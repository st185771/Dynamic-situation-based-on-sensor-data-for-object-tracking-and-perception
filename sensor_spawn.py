import carla
import pygame
import numpy as np
import cv2
from ultralytics import YOLO
import random
from sort import Sort

# Initialize CARLA client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Set adverse weather conditions
weather = carla.WeatherParameters(
    cloudiness=80.0,
    precipitation=80.0,
    sun_altitude_angle=30.0
)
world.set_weather(weather)

# Function to convert CARLA image to a format suitable for YOLO
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # RGBA to RGB
    array = array[:, :, ::-1]  # RGB to BGR
    return array

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Initialize Pygame
pygame.init()
display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
font = pygame.font.SysFont('Arial', 20)

# Function to spawn a vehicle
def spawn_vehicle():
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        vehicle.set_autopilot(True, traffic_manager.get_port())
    return vehicle

def main():
    actor_list = []
    vehicle_velocities = {}
    ego_speed_text = ""

    try:
        # Add ego vehicle
        ego_vehicle = spawn_vehicle()
        if not ego_vehicle:
            print("Failed to spawn ego vehicle.")
            return
        actor_list.append(ego_vehicle)

        # Add other vehicles
        for i in range(5):
            vehicle = spawn_vehicle()
            if vehicle:
                actor_list.append(vehicle)

        # Add a camera sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        actor_list.append(camera)

        # Add a radar sensor
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '30')
        radar_bp.set_attribute('range', '20')
        radar_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        radar = world.spawn_actor(radar_bp, radar_transform, attach_to=ego_vehicle)
        actor_list.append(radar)

        # Initialize SORT tracker
        tracker = Sort()

        last_measurement_time = world.get_snapshot().timestamp.elapsed_seconds

        def radar_callback(radar_data):
            nonlocal last_measurement_time, vehicle_velocities

            current_time = radar_data.timestamp
            time_diff = current_time - last_measurement_time
            last_measurement_time = current_time

            for detection in radar_data:
                velocity_kmh = detection.velocity * 3.6  # Convert m/s to km/h
                azimuth = detection.azimuth
                altitude = detection.altitude
                distance = detection.depth
                object_id = (azimuth, altitude, distance)
                vehicle_velocities[object_id] = velocity_kmh

        radar.listen(radar_callback)

        def camera_callback(image):
            nonlocal ego_vehicle, vehicle_velocities, ego_speed_text

            img = process_image(image)
            img_writable = img.copy()
            results = model(img)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf])
            
            tracks = tracker.update(np.array(detections))

            for track in tracks:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                velocity = vehicle_velocities.get(track_id, 0)
                cv2.rectangle(img_writable, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_writable, f'ID: {int(track_id)}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(img_writable, f'Vel: {velocity:.2f} km/h', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            ego_velocity = ego_vehicle.get_velocity()
            ego_speed = np.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2) * 3.6  # Convert m/s to km/h
            ego_speed_text = f'Ego Vehicle Velocity: {ego_speed:.2f} km/h'
            cv2.putText(img_writable, ego_speed_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            surface = pygame.surfarray.make_surface(cv2.flip(cv2.rotate(img_writable, cv2.ROTATE_90_CLOCKWISE), 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()

        camera.listen(camera_callback)

        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.75))
                    elif event.key == pygame.K_s:
                        ego_vehicle.apply_control(carla.VehicleControl(brake=1.0))
                    elif event.key == pygame.K_a:
                        ego_vehicle.apply_control(carla.VehicleControl(steer=-0.5))
                    elif event.key == pygame.K_d:
                        ego_vehicle.apply_control(carla.VehicleControl(steer=0.5))
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_w, pygame.K_s]:
                        ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
                    elif event.key in [pygame.K_a, pygame.K_d]:
                        ego_vehicle.apply_control(carla.VehicleControl(steer=0.0))

            world.tick()
            clock.tick(60)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        for actor in actor_list:
            if actor.is_alive:
                actor.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
