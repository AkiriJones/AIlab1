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
    def __init__(self, parent=None):
        self.position = None
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

def heuristic(curr,goal):
    currDiff = difficultyMap[colorCoords[int(curr[0]),int(curr[1])]]
    # value = (abs(goal[0] - int(curr[0])) + abs(goal[1] - int(curr[1])))
    # value = value * currDiff
    return max((abs(goal[0] - int(curr[0])) , abs(goal[1] - int(curr[1])))) * currDiff


def astar(start,checkList:list[list[str]], total_meters:int):
    """Finds the shortest path using the A* algorithm."""
    open_list = []
    closed_set = set()
    visited_points = []
    start_node = Node(None)
    start_node.setPosition(start.getPosition())
    visited_points.append(start)
    end_node = Node(None)
    end_coords = [int(checkList[0][0]), int(checkList[0][1])]
    end_node.setPosition(end_coords)

    heapq.heappush(open_list, start_node)
    checkList.remove(checkList[0])
    counter = 0
    while open_list:
        current_node = heapq.heappop(open_list)  # Get node with lowest f-score
        # print("Points to hit " + str(checkList))
        # print("current: " + str(current_node.getPosition()))
        # print()

        closed_set.add(tuple(current_node.position))
        # print()
        if isgoalpoint(current_node.position,checkList):
            # print("point " + str(checkList[0]) + "reached")
            checkList.remove(checkList[0])
            visited_points.append(current_node)
            closed_set.clear()
            if checkList:
                end_coords = [int(checkList[0][0]), int(checkList[0][1])]
                end_node.setPosition(end_coords)

        if isgoal(visited_points):
            path = []
            while current_node:
                path.append(current_node.position)
                if current_node != start_node:
                    x1,x2 = current_node.getPosition()[0], current_node.parent.getPosition()[0]
                    y1,y2 = current_node.parent.getPosition()[1], current_node.parent.getPosition()[1]
                    abs_x = abs(x1 - x2)
                    abs_y = abs(y1 - y2)
                    total_meters += (abs_x * xinmeters) + (abs_y * yinmeters)
                current_node = current_node.parent

            return path[::-1],total_meters  # Return reversed path and total meters traversed


        neighbors = checkNeighbors(current_node.position[0], current_node.position[1])
        neighbors = getBestNeighbor(closed_set, current_node,neighbors,checkList)
        for neighbor in neighbors: # Possible moves
            nbrCoords = neighbor
            neighbor = Node(current_node)
            neighbor.setPosition(nbrCoords)
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor.position, end_node.position)
            neighbor.f = neighbor.g + neighbor.h
            if any(n.position == neighbor.position and n.g <= neighbor.g for n in open_list):
                continue
            if tuple(neighbor.position) not in closed_set:
                heapq.heappush(open_list, neighbor)
    return None


def isgoal(n)-> bool:
    copyPath = goalPath.copy()
    currPath = n.copy()
    for coord in currPath:
        if coord.position in copyPath:
            copyPath.remove(coord)
    if copyPath:
        return False
    return True

def isgoalpoint(coords,checkList: list[list[str]])-> bool:
    if not checkList:
        return False
    else:
        point = checkList[0]
        return abs(coords[0] - int(point[0])) <= 1 and abs(coords[1] - int(point[1])) <= 1

def getElevations(filename) -> dict:
    elevationCoords = {}
    file = open(filename, 'r')
    y = 0
    for line in file:
        if y <= 494:
            elevations = line.strip().split()
            for x in range(395):
                coord = [x,y]
                elevationCoords[tuple(coord)] = elevations[x]
            # print(elevations)
        y += 1
    file.close()
    return elevationCoords

def getDifficulties(colorset) -> dict:
    difficulties = {}
    for color in colorset:
            match color:
                case (2,208,60): #Slow run forest
                    difficulties[color] = 0.5
                case (255,192,0): #Rough meadow
                    difficulties[color] = 0.7
                case (0,0,0): #Footpath
                    difficulties[color] = 0.01
                case (5, 73, 24): #Impassible vegetation
                    difficulties[color] = 0.95
                case (255, 255, 255): #Easy movement forest
                    difficulties[color] = 0.05
                case (71, 51, 3): #Paved road
                    difficulties[color] = 0.01
                case (205, 0, 101): #Out of Bounds
                    difficulties[color] = 1
                case (2, 136, 40): #Walk forest
                    difficulties[color] = 0.5
                case (248, 148, 18): #Open land
                    difficulties[color] = 0.01
                case (0, 0, 255): #Lake/Swamp/Marsh
                    difficulties[color] = 0.9
                case _:
                    difficulties[color] = 0
    return difficulties

def getPath(filename):
    path = []
    file = open(filename, 'r')
    for line in file:
        path.append(line.split())
    return path
def checkNeighbors(oldX:str, oldY:str) -> list[list[int]]:
    neighbors = []
    oldX = int(oldX)
    oldY = int(oldY)
    for x in range(oldX-1, oldX+2):
        y = oldY
        for y in range(oldY - 1, oldY + 2):
            if not (x < 0) | (x > 395):
                if not (y < 0) | (y > 500):
                    if not (x == oldX and y == oldY):
                        neighbors.append([x, y])
    return neighbors

def getBestNeighbor(currlist: set[tuple[list[int]]],curr:Node, neighbors:list[list[int]], checklist:list[list[str]]) -> list[list[int]]:
    point = checklist[0]
    point_x = int(point[0])
    point_y = int(point[1])
    currX_abs = abs(curr.position[0] - point_x)
    currY_abs = abs(curr.position[1] - point_y)

    nbrs = []
    for nbr in neighbors:
        if not difficultyMap[colorCoords[nbr[0],nbr[1]]] == 1:
            if not currlist.__contains__(tuple(nbr)):
                nbrX_abs = abs(nbr[0] - point_x)
                nbrY_abs = abs(nbr[1] - point_y)
                if nbrX_abs < currX_abs or nbrY_abs < currY_abs:
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
        elevationCoords = getElevations(elevation)
        path_filename = args[2]
        output_filename = args[3]
        image = Image.open(image)
        # image.show()
        pixels = image.load()
        # 395 rows of 500 cols representing each pixel's color on a graph.
        colorCoords = {}
        coords = set()
        colorSet = set()
        for x in range(395):
            for y in range(500):
                coord = [x,y]
                coords.add(tuple(coord))
                coordColor = pixels[x, y][:-1]
                colorCoords[tuple(coord)] = coordColor
                colorSet.add(coordColor)
        path = getPath(path_filename)
        # 1 being the easiest, 0 being the hardest
        difficultyMap = getDifficulties(colorSet)
        goalPath = path.copy()
        # goalPath.remove(goalPath[0])
        startNode = Node(None)
        startcoords = [int(path[0][0]),int(path[0][1])]
        startNode.setPosition(startcoords)
        traversedPath, total_meters = astar(startNode,goalPath,total_meters)
        drawing_path = []
        for point in traversedPath:
            drawing_path.append(tuple(point))
        path_drawing = ImageDraw.Draw(image)
        path_drawing.line(drawing_path, fill='#a146dd', width=1)
        print("Total distance: " + str(total_meters))
        # print("Done!")
        # print("Path: " + str(traversedPath))
        # image.save(output_filename)
        image.show()

        # print(traversedPath)