import math
import random
import sys
from collections import deque, defaultdict

from sys import argv
from PIL import Image, ImageDraw
import heapq

xinmeters = 10.29
yinmeters = 7.55
total_meters = 0


class Node:
    def __init__(self, parent=None, coordinates=None):
        self.position = coordinates
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f  # Priority queue sorting

    def getPosition(self) -> list[int]:
        return self.position

    def setPosition(self, position: list[int]) -> None:
        self.position = position


def heuristic(curr, goal):
    currDiff = difficultyMap.get(colorCoords.get((int(curr[0]), int(curr[1]))), 1)
    # change_in_elevation = abs(elevationCoords[curr[0], curr[1]] - elevationCoords[goal[0], goal[1]])
    return min((abs(goal[0] - curr[0]), abs(goal[1] - curr[1]))) * currDiff


def getTotalDistance(path: list[list[int]]) -> int:
    path = path[::-1]
    total_meters = 0
    for i in range(len(path) - 1):
            x1, x2 = path[i][0], path[i + 1][0]
            y1, y2 = path[i][1], path[i + 1][1]
            abs_x = abs(x1 - x2) * xinmeters
            abs_y = abs(y1 - y2) * yinmeters
            total_meters += math.sqrt(abs_x ** 2 + abs_y ** 2)
    return total_meters

def astar(start: Node, checkPoint: list[int]) -> list[Node] | None:
    """Finds the shortest path using the A* algorithm."""
    open_list = []
    closed_set = set()
    start_node = Node(None,start)

    heapq.heappush(open_list, start_node)
    while open_list:
        current_node = heapq.heappop(open_list)  # Get node with lowest f-score
        # currX = current_node.getPosition()[0]
        # currY = current_node.getPosition()[1]
        # if list[(231,326)] == list[(currX, currY)] and checkPoint[0] == 230 and checkPoint[1] == 327:
        #     print()
        # print("Points to hit " + str(checkList))
        # print("current: " + str(current_node.getPosition()))
        # print()

        closed_set.add(tuple(current_node.position))
        if isgoalpoint(current_node.position, checkPoint):
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        neighbors = checkNeighbors(current_node.position[0], current_node.position[1])
        neighbors = getBestNeighbor(closed_set, current_node, neighbors, checkPoint)
        for neighbor in neighbors:  # Possible moves
            nbrCoords = neighbor
            neighbor = Node(current_node, nbrCoords)
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor.position, checkPoint)
            neighbor.f = neighbor.g + neighbor.h
            if any(n.position == neighbor.position and n.g <= neighbor.g for n in open_list):
                continue
            if tuple(neighbor.position) not in closed_set:
                heapq.heappush(open_list, neighbor)
    return None


def ConstructPath(start, waypoints, goal):
    total_path = []
    current_position = start.getPosition()
    for waypoint in waypoints:  # Move through each intermediate goal
        if current_position[0] == 303 and  current_position[1] == 240:
            print()
        path_segment = astar(current_position,waypoint)
        if path_segment is None:
            return None  # No valid path exists

        total_path.extend(path_segment[:-1])  # Avoid duplicate points
        current_position = waypoint  # Move to the next stage

    total_path.append(goal.position)  # Add final goal
    return total_path



def isgoal(n) -> bool:
    copyPath = goalPath.copy()
    currPath = n.copy()
    for coord in currPath:
        if coord.position in copyPath:
            copyPath.remove(coord)
    if copyPath:
        return False
    return True


def isgoalpoint(coords, checkpoint: list[int]) -> bool:
    if not checkpoint:
        return False
    else:
        return abs(coords[0] - checkpoint[0]) < 1 and abs(coords[1] - checkpoint[1]) < 1


def getElevations(filename) -> dict:
    elevationCoords = {}
    file = open(filename, 'r')
    y = 0
    for line in file:
        if y <= img_height - 1:
            elevations = line.strip().split()
            for x in range(int(img_width)):
                coord = [x, y]
                elevationCoords[tuple(coord)] = float(elevations[x])
        y += 1
    file.close()
    return elevationCoords


def getDifficulties(colorset) -> dict:
    difficulties = {}
    for color in colorset:
        match color:
            case (2, 208, 60):  #Slow run forest
                difficulties[color] = 0.5
            case (255, 192, 0):  #Rough meadow
                difficulties[color] = 0.7
            case (0, 0, 0):  #Footpath
                difficulties[color] = 0.01
            case (5, 73, 24):  #Impassible vegetation
                difficulties[color] = 0.95
            case (255, 255, 255):  #Easy movement forest
                difficulties[color] = 0.05
            case (71, 51, 3):  #Paved road
                difficulties[color] = 0.01
            case (205, 0, 101):  #Out of Bounds
                difficulties[color] = 1
            case (2, 136, 40):  #Walk forest
                difficulties[color] = 0.5
            case (248, 148, 18):  #Open land
                difficulties[color] = 0.01
            case (0, 0, 255):  #Lake/Swamp/Marsh
                difficulties[color] = 0.9
            case (191,191,191):
                difficulties[color] = 1
            case (175, 80, 122):
                difficulties[color] = 1
            case _:
                difficulties[color] = 100
    return difficulties


def getPath(filename):
    path = []
    file = open(filename, 'r')
    for line in file:
        line = line.split()
        xy = []
        for nums in line:
            xy.append(int(nums))
        path.append(xy)
    return path


def checkNeighbors(oldX: int, oldY: int) -> list[list[int]]:
    neighbors = []
    xArray = [oldX-1,oldX,oldX+1]
    yArray = [oldY-1,oldY,oldY+1]
    for x in xArray:
        for y in yArray:
            if not (x < 0) | (x >= 395):
                if not (y < 0) | (y >= 500):
                    if not (x == oldX and y == oldY):
                        neighbors.append([x, y])
    return neighbors


def getBestNeighbor(currlist: set[tuple[list[int]]], curr: Node, neighbors: list[list[int]],
                    checkpoint: list[int]) -> list[list[int]]:
    point_x = checkpoint[0]
    point_y = checkpoint[1]
    currX_abs = abs(curr.position[0] - point_x)
    currY_abs = abs(curr.position[1] - point_y)
    curr_Ele = elevationCoords[(point_x, point_y)]
    nbrs = []
    for nbr in neighbors:
        nbr_ele = elevationCoords[(nbr[0], nbr[1])]
        change = abs(nbr_ele - curr_Ele)
        if change <= 25:
            if difficultyMap[colorCoords[nbr[0], nbr[1]]] < 1:
                if not currlist.__contains__(tuple(nbr)):
                    # nbrX_abs = abs(nbr[0] - point_x)
                    # nbrY_abs = abs(nbr[1] - point_y)
                    # if nbrX_abs < currX_abs or nbrY_abs < currY_abs:
                    nbrs.append(nbr)
    return nbrs


if __name__ == '__main__':
    args = argv[1:]
    if len(args) != 4:
        print("Usage: python lab1.py terrain-image elevation-file path-file output-image-filename", file=sys.stderr)
        # sys.exit(1)
    else:
        image = args[0]
        elevation = args[1]
        path_filename = args[2]
        output_filename = args[3]
        map = Image.open(image)
        map.save(image)
        map.convert('RGB')
        img_width, img_height = map.size
        elevationCoords = getElevations(elevation)
        # image.show()
        pixels = map.load()
        # 395 rows of 500 cols representing each pixel's color on a graph.
        colorCoords = {}
        coords = set()
        colorSet = set()
        for x in range(int(img_width)):
            for y in range(int(img_height)):
                coord = [x, y]
                coords.add(tuple(coord))
                coordColor = pixels[x, y][:-1]
                colorCoords[tuple(coord)] = coordColor
                colorSet.add(coordColor)
        path = getPath(path_filename)
        # 1 being the easiest, 0 being the hardest
        difficultyMap = getDifficulties(colorSet)
        goalPath = path.copy()
        goalPath.remove(goalPath[0])
        startcoords = [path[0][0], path[0][1]]
        startNode = Node(None, startcoords)
        endcoords = [path[len(path)-1][0], path[len(path)-1][1]]
        endNode = Node(None, endcoords)
        traversedPath = ConstructPath(startNode, goalPath, endNode)
        if traversedPath:
            print(getTotalDistance(traversedPath))
            drawing_path = []
            for point in traversedPath:
                drawing_path.append(tuple(point))
            path_drawing = ImageDraw.Draw(map)
            path_drawing.line(drawing_path, fill='#a146dd', width=1)
            map.save(output_filename, format='PNG')
            map.show()
        else:
            print("Couldn't find path.")
            map.save(output_filename, format='PNG')
            map.show()