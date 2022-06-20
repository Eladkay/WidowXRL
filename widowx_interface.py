from rl_project.widowx import GenericWidowX


class WidowXInterface(GenericWidowX):

    def __init__(self, widowx):
        """
        :param widowx: the WidowX object to use, must have move_to_xyz implemented!
        """
        self.widowx = widowx
        self.location = (0, 0)
        self.gripper_open = True

        # open camera
        self.widowx.open_camera()

        # call reset to move to neutral position
        self.reset()

    def step(self, location):
        """
        Moves the WidowX to the given location.
        Warning: will not throw an exception if the parameter is not a valid location.
        :return: True if the move was successful, False otherwise.
        """
        return self.move_to_location(*location)

    def get_pos(self):
        return self.location

    def reset(self):
        # move to neutral position
        self.location = (0, 0)
        self.widowx.move_to_xyz(*self.location)
        # reset gripper
        self.gripper_open = True
        self.widowx.open_gripper()

    def get_image(self):
        location = self.location
        self.move_to_location(0, 0)
        image = self.widowx.get_image()
        self.move_to_location(*location)
        return image

    def get_binary_image(self):
        """
        Returns a binary image of the current camera image.
        :return: a binary image with red objects as 1 and everything else as 0
        """
        location = self.location
        self.move_to_location(0, 0)
        img = _self.detect_red(self.widowx.parking_manager.img)
        w, h = img.shape
        for i in range(w):
            for j in range(h):
                if img[i][j]:
                    img[i][j] = 1

        image = np.array(img)
        self.move_to_location(*location)
        return image

    def grab_cube(self):
        """
        Tries to grab a cube at the current location.
        """
        self.widowx.open_gripper()
        self.widowx.move_to_grasp(self.location[0], self.location[1], 0.44, 0)
        self.widowx.close_gripper()
        self.gripper_open = False
        self.widowx.move_to_grasp(self.location[0], self.location[1], 0.4, 0)

    def move_to_location(self, x, y):
        """
        Moves the WidowX to the given location.
        :return: True if the move was successful, False otherwise.
        """
        if self.location == (x, y):
            return True

        self.location = (x, y)

        if self.location == (0, 0):
            return self.widowx.move_to_neutral()

        return self.widowx.move_to_xyz(x, y)

    def drop_cube(self):
        """
        Drops the cube the gripper is holding.
        """
        if self.gripper_open:
            print("Warning: called drop but gripper is already open")

        self.widowx.move_to_grasp(self.location[0], self.location[1], 0.44, 0)
        self.widowx.open_gripper()
        self.gripper_open = True
        return self.widowx.move_to_grasp(self.location[0], self.location[1], 0.4, 0)

    @staticmethod
    def _detect_red(img):
        """
        Detects red objects in the image
        :return: a binary image with red objects as 1 and everything else as 0
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask = mask0 + mask1

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        output = np.zeros(mask.shape)
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 150:
                cv2.drawContours(output, [cnt], -1, 255, -1)

        return output


class WidowX_wrapper:
    def __init__(self, widowx, debug=True):
        self.widowx = widowx
        self.Debug = debug

        # move to neutral position
        self.location = (0, 0)
        self.widowx.move_to_xyz(*self.location)
        self.widowx.open_camera()

        # reset gripper
        self.gripper_open = True
        self.widowx.open_gripper()

    def safe_print(self, text):
        if self.Debug:
            print("--- " + text)

    def grab_cube(self):
        self.widowx.open_gripper()
        self.widowx.move_to_grasp(self.location[0], self.location[1], 0.44, 0)
        self.widowx.close_gripper()
        self.gripper_open = False
        return self.widowx.move_to_grasp(self.location[0], self.location[1], 0.4, 0)

    def move_to_location(self, x, y):
        if self.location == (x, y):
            return True

        self.safe_print("going to location: " + str((x, y)))
        self.location = (x, y)

        if self.location == (0, 0):
            return self.widowx.move_to_neutral()

        return self.widowx.move_to_xyz(x, y)

    def drop(self):
        if self.gripper_open:
            print("Warning: called drop but gripper is already open")

        self.safe_print("dropping cube")
        self.widowx.move_to_grasp(self.location[0], self.location[1], 0.44, 0)
        self.widowx.open_gripper()
        self.gripper_open = True
        return self.widowx.move_to_grasp(self.location[0], self.location[1], 0.4, 0)

    def grab_and_eval(self):
        self.grab_cube()
        self.widowx.move_to_eval()
        parking_place, _ = self.widowx.parking_manager.get_parking_place(self.widowx.classifier)
        is_cube_in_gripper = parking_place != -1
        self.safe_print("Gripper evaluation: " + str(is_cube_in_gripper))
        if not is_cube_in_gripper:
            self.widowx.open_gripper()
            self.gripper_open = True

        return is_cube_in_gripper

    def move_grab_and_eval(self, x, y):
        self.move_to_location(x, y)
        return self.grab_and_eval()

    def get_img(self):
        self.move_to_location(0, 0)
        img = red(self.widowx.parking_manager.img)
        w, h = img.shape
        for i in range(w):
            for j in range(h):
                if img[i][j]:
                    img[i][j] = 1

        return np.array(img)
