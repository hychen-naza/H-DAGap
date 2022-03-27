from operator import attrgetter

class LaserData:
    def __init__(self, depth: float, rel_x: float, rel_y: float):
        self.x = rel_x
        self.y = rel_y
        self.depth = depth


laser_datas = [LaserData(3, 1,2), LaserData(4, 5,6), LaserData(1,0.2,2), LaserData(-3,2,2)]
min_laser = min(laser_datas, key=attrgetter('depth'))
print(min_laser.depth, min_laser.x, min_laser.y)