import time, random, carla, cv2, math
import numpy as np

"""https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/
   https://carla.readthedocs.io/en/latest/python_api_tutorial/

   A good portion of this code comes from the links above"""

class CarlaEnvironment():

    img_width, img_height = 600,400
    wait_camera = None
    return_camera = None
    limit_time = 60
    show_cam = False

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter('model3')[0]

    def reset(self):
        """Vehicle"""
        self.actor_list, self.collision_sensor_list = [], []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.transform)
        self.actor_list.append(self.vehicle)

        """Camera"""
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '600')
        self.camera_bp.set_attribute('image_size_y', '400')
        self.camera_bp.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.rgb_sensor = self.world.spawn_actor(self.camera_bp, transform,
                                                attach_to=self.vehicle)
        self.actor_list.append(self.rgb_sensor)
        self.rgb_sensor.listen(lambda data: self.preprocess_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        """Collision Sensor"""
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor,
                                                transform,
                                                attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        if self.wait_camera is None:
            time.sleep(0.1)

        self.time_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))


        return self.return_camera

    def collision_data(self, event):
        self.collision_sensor_list.append(event)

    def preprocess_img(self, image):
        i = np.array(image.raw_data)
        				#Y,X,RGB
        i2 = i.reshape((600, 400, 4))
        i3 = i2[:, :, :3]
        if self.show_cam:
            cv2.imshow("window", i3)
            cv2.waitKey(1)
        self.return_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*1.0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*1.0))
        #https://stackoverflow.com/questions/48943163/c-convert-3d-velocity-vector-to-speed-value
        car_speed = self.vehicle.get_velocity()
        car_speed = int(3.6 * math.sqrt(car_speed.x**2 + car_speed.y**2 + car_speed.z**2))
        if len(self.collision_sensor_list) != 0:
            done = True
            reward = -100
        elif car_speed < 40:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.time_start + 60 < time.time():
            done = True


        return self.return_camera, reward, done, None
