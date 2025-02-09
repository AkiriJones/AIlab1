import random
import sys
from collections import deque, defaultdict


from sys import argv
from PIL import Image
import heapq
goal = list()


class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f  # Priority queue sorting

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


def aStarSearch(start,checkList: list[list[str]]):
    visited = []
    queue = deque([start])
    path = defaultdict(list)
    path[None] = start
    checkPath = checkList.copy()
    checkPath.remove(start)
    while queue:
        node = queue.popleft()
        if isgoal(path):
            return path
        else:
            if isgoalpoint(node,checkPath):
                checkPath.remove(checkPath[0])
            visited.append(node)
            #variable to get the most optimal neighbor
            if node[0] == 276 and node[1] == 279:
                print()
            nbrs = checkNeighbors(node[0], node[1])
            nbrs = getBestNeighbor(node, nbrs, checkPath)

            nbrDiffs = list()
            for nbr in nbrs:
                nbrDiffs.append(difficultyMap[colorCoords[nbr[0],nbr[1]]])
            chosenNbr = random.Random().choices(nbrs, weights = nbrDiffs, k = 1)
            path[tuple(node)] = chosenNbr[0]
            queue.append(chosenNbr[0])
    return None

#Get the neighbors that gets us closer to the next checkpoint in the list
def getBestNeighbor(curr: list[str],neighbors:list[list[int]],checklist:list[list[str]]) -> list[list[int]]:
    point = checklist[0]
    nbrs = []
    for nbr in neighbors:
        if ( int(curr[0]) < nbr[0] <= int(point[0]))  | (int(curr[1]) < nbr[1] <= int(point[1])):
            nbrs.append(nbr)
        elif int(point[0]) <= nbr[0] < int(curr[0]) | int(point[1]) <= nbr[1] < int(curr[1]):
            nbrs.append(nbr)
    return nbrs




#Checks if all coords in the path in the goal list
def isgoal(n)-> bool:
    copyPath = goalPath.copy()
    for coord in n:
        if n[coord] in copyPath:
            copyPath.remove(n[coord])
    if copyPath:
        return False
    return True

def isgoalpoint(coords,checkList: list[list[str]])-> bool:
    point = checkList[0]
    return coords[0] == int(point[0]) and coords[1] == int(point[1])



def getPath(filename):
    path = []
    file = open(filename, 'r')
    for line in file:
        path.append(line.split())
    return path

def getElevations(filename) -> dict:
    elevationCoords = {}
    file = open(filename, 'r')
    y = 0
    for line in file:
        if y <= 494:
            elevations = line.strip().split()
            for x in range(395):
                coords = [x,y]
                elevationCoords[tuple(coords)] = elevations[x]
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
                    difficulties[color] = 0
                case (5, 73, 24): #Impassible vegetation
                    difficulties[color] = 0.95
                case (255, 255, 255): #Easy movement forest
                    difficulties[color] = 0.05
                case (71, 51, 3): #Paved road
                    difficulties[color] = 0
                case (205, 0, 101): #Out of Bounds
                    difficulties[color] = 1
                case (2, 136, 40): #Walk forest
                    difficulties[color] = 0.5
                case (248, 148, 18): #Open land
                    difficulties[color] = 0
                case (0, 0, 255): #Lake/Swamp/Marsh
                    difficulties[color] = 0.9
                case _:
                    difficulties[color] = 0
    return difficulties

if __name__ == '__main__':
    args = argv[1:]
    if len(args) != 4:
        print("Usage: python lab1.py terrain-image elevation-file path-file output-image-filename", file=sys.stderr)
        # sys.exit(1)
    else:
        image = args[0]
        elevation = args[1]
        elevationCoords = getElevations(elevation)
        # print(int(float(elevationCoords[394,494])))
        path_filename = args[2]
        output_filename = args[3]
        xinmeters = 10.29
        yinmeters = 7.55
        image = Image.open(image)
        # image.show()
        pixels = image.load()
        # 395 rows of 500 cols representing each pixel's color on a graph.
        colorCoords = {}
        colorSet = set()
        for x in range(395):
            for y in range(500):
                coords = [x,y]
                coordColor = pixels[x, y][:-1]
                colorCoords[tuple(coords)] = coordColor
                colorSet.add(coordColor)
        path = getPath(path_filename)
        # 1 being the easiest, 0 being the hardest
        difficultyMap = getDifficulties(colorSet)
        goalPath = path.copy()
        # goalPath.remove(goalPath[0])
        # nbrs = checkNeighbors(230,326)
        # print(nbrs)
        # print(getBestNeighbors(['230','326'],nbrs,goalPath))
        traversedPath = aStarSearch(path[0], path)
        print(traversedPath)
        # print(difficultyMap)
        # print(checkNeighbors(int(path[0][0]), int(path[0][1])))
        # print(checkNeighbors(30,0))




        # for color in colorCoords:
        #     print(colorCoords[color])
        # for elevation in elevationCoords:
        #     print(elevationCoords[elevation])
