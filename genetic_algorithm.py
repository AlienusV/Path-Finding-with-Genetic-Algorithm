"""random for chance calculations
numpy for vector and array calculations
time for waiting
datetime for naming saveFile
threading for simultaneous populations, collision calculation and screen
pygame for screen
colorama for colored output
xlwt for saving in xls format
schedule for saving every minute"""
import random
import numpy as np
import time
import datetime
import threading
from threading import Thread
import pygame
import colorama
import xlwt
import xlrd
from xlutils.copy import copy as xlcopy
from ast import literal_eval as make_tuple
import schedule
import os
import shutil

"""testing toggle"""
testing = False
backgroundOnly = False
load = False
loadFrom = "./2022-10-18 21:33:32.338539/"


"""creating saveFile"""
if load:
    loadExcel = xlrd.open_workbook(loadFrom + "Populations.xls")
    dataset = xlcopy(loadExcel)
    datasetName = (loadFrom + "Populations.xls")
    print(loadExcel, dataset)
else:
    startDate = datetime.datetime.now()
    os.mkdir("./" + str(startDate))
    dataset = xlwt.Workbook()
    datasetName = ("./" + str(startDate) + "/Populations.xls")

"""setting pygame display"""
windowWidth = 800
windowHeight = 800
pygame.init()
screen = pygame.display.set_mode([windowWidth, windowHeight])
pygame.display.set_caption('Genetic Algorithm Learning')

"""setting colors"""
colorama.init()
white = (255, 255, 255)
pink = (255, 20, 147)
red = (255, 0, 0)
orange = (255, 165, 0)
yellow = (255, 255, 0)
green = (0, 255, 0)
cyan = (0, 255, 255)
blue = (0, 0, 255)
magenta = (255, 0, 255)
indigo = (75, 0, 130)
navy = (0, 0, 128)
brown = (165, 42, 42)
darkGreen = (0, 100, 0)
gray = (150, 150, 150)
gray2 = (155, 155, 155)
black = (0, 0, 0)

"""setting thread stop events"""
stopEvent1 = threading.Event()
stopEvent2 = threading.Event()

stopEvent3 = threading.Event()
stopEvent4 = threading.Event()
stopEvent5 = threading.Event()
stopEvent6 = threading.Event()
stopEvent7 = threading.Event()
stopEvent8 = threading.Event()
stopEvent9 = threading.Event()
stopEvent10 = threading.Event()
stopEvent11 = threading.Event()

"""declaring where to go"""
goal = np.array((775, 35))

"""lists for iterating stuff """
boxes = []
rewards = []
populations = []
sheets = []

colorList = {
    (255, 0, 0): colorama.Fore.RED,
    (0, 255, 0): colorama.Fore.GREEN,
    (255, 255, 0): colorama.Fore.YELLOW,
    (0, 0, 255): colorama.Fore.BLUE,
    (0, 255, 255): colorama.Fore.CYAN,
    (0, 0, 0): colorama.Fore.BLACK,
    (255, 0, 255): colorama.Fore.MAGENTA
}


class Brain:
    """Brain: Remembering how to move"""

    def __init__(self, size, mutationRate, brain=None):
        self.size = size
        self.step = 0
        self.mutationRate = mutationRate
        if load:
            self.directions = brain
        else:
            self.directions = np.ones((self.size, 2))
            self.randomize()

    def randomize(self):
        """fills the brain with random data"""
        for i in range(self.size):
            randomDirection = np.array((random.uniform(-1, 1), random.uniform(-1, 1)))
            self.directions[i, :] = randomDirection

    def clone(self):
        clone = Brain(self.size, self.mutationRate)
        clone.directions = self.directions
        return clone

    def mutate(self):
        """go through every single brainstep and mutate it if it meets the mutationRate"""
        for i in range(self.size):
            chance = random.uniform(0, 1)
            if chance < self.mutationRate:
                randomDirection = np.array((random.uniform(-1, 1), random.uniform(-1, 1)))
                self.directions[i, :] = randomDirection


class Box:
    """Red box, that limits the path."""

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        boxes.append(self)

    def draw(self):
        pygame.draw.rect(screen, (255, 0, 0), [self.x, self.y, self.width, self.height])


class Reward:
    """Green Reward, shows the way that dots should go"""

    def __init__(self, x, y, width=10, height=10):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.midpoint = np.array((self.x + self.width / 2, self.y + self.height / 2))
        rewards.append(self)

    def draw(self):
        pygame.draw.rect(screen, gray2, [self.x, self.y, self.width, self.height])


class Dot:
    """Dot, member of a population, Color1 dictates what color a normal member is. Color2 dictates the best"""

    def __init__(self, brainSize, color1, color2, shape, mutationRate, decisionRate, maxVel, brain=None, beingBorn=None):
        """pos acc and vel determine movement vectors
        isBest, reachedGoal, dead determines the state of the Dot
        mutationRate and decisionRate determines the intelligent and how good they can reproduce
        fitness and reward_punishment determines how good they've done
        rewards dictionary and rewardCount help to determine which reward has been taken and which to go next"""
        self.pos = np.array([400, 780])
        self.acc = np.array([0, 0])
        self.vel = np.array([0, 0])
        self.maxVel = maxVel
        self.isBest = False
        self.reachedGoal = False
        self.dead = False
        self.mutationRate = mutationRate
        self.decisionRate = decisionRate
        self.fitness = 0
        self.reward_punishment = 0
        self.color1 = color1
        self.color2 = color2
        self.shape = shape
        self.rewards = {}
        self.rewardCount = 0
        for reward in rewards:
            self.rewards[reward] = 0
        if load:
            self.brain = Brain(brainSize, self.mutationRate, brain)
        else:
            if not beingBorn:
                self.brain = Brain(brainSize, self.mutationRate)

    def draw(self, screen1):
        """if normal member, draws just itself in color1, if best member, draws itself in color2 and draws a line
        between next reward and itself"""
        if self.isBest:
            shapes(self.pos, self.color2, self.shape)
            if self.rewardCount == len(self.rewards):
                pygame.draw.line(screen1, self.color2, goal, self.pos)
            else:
                pygame.draw.line(screen1, self.color2,
                                 (rewards[np.clip(self.rewardCount - 1, 0, len(self.rewards) - 1)]).midpoint, self.pos)
        else:
            shapes(self.pos, self.color1, self.shape)
            # pygame.draw.line(screen1,self.color1,
            #                 (rewards[np.clip(self.rewardCount, 0, len(self.rewards) - 1)]).midpoint, self.pos)
            pygame.draw.line(screen1,self.color1,self.pos, self.pos + (self.acc * 20))
            pygame.draw.line(screen1,self.color2,self.pos, self.pos + (self.vel * 10))

    def distanceTo(self, objPos):
        """determines the distance between itself and target obj, returns the difference vector and distance"""
        dist1 = objPos - self.pos
        dist2 = np.dot(dist1.T, dist1)
        dist3 = np.sqrt(dist2)
        return dist1, dist3

    def move(self):
        """chance determines if it can 'decide' or let the 'memory' from its past determine its next move
        calculating the 'decision' is done by taking the difference vector and dividing it by the distance, thus getting
        the unit vector towards to next goal. after this decision it 'saves' in its brain to 'remember' it for the next
        generation. np.clip slows the movement"""
        self.acc = np.array((0, 0))
        chance = random.uniform(0, 1)
        if self.brain.size > self.brain.step:
            if chance > self.decisionRate:
                self.acc = self.brain.directions[self.brain.step, :]
                self.brain.step += 1
            else:
                difference, distance = self.distanceTo(
                    (rewards[np.clip(self.rewardCount, 0, len(self.rewards) - 1)]).midpoint)
                pointingVector = difference / distance
                self.brain.directions[self.brain.step, :] = pointingVector
                self.acc = pointingVector
                self.brain.step += 1
        else:
            self.dead = True
        self.vel = self.vel + self.acc
        self.vel = np.clip(self.vel, -1 * self.maxVel, self.maxVel)
        self.pos = self.pos + self.vel

    def boxcollisions(self):
        """function to determine collisions, if it hits a box it gets X amount of punishment,
        if it hits(goes through) a reward, it gains Y amount of reward points. Hitting a reward makes the rewardCount
        go up and marks that reward in its rewards dictionary as 'taken'(1)"""
        if not self.dead:
            for box in boxes:
                if box.x < self.pos[0] < box.x + box.width \
                        and box.y < self.pos[1] < box.y + box.height:
                    self.dead = True
                    self.reward_punishment += -0.1

    def rewardcollisions(self):
        if not self.dead:
            for i in (range(self.rewardCount - 2, self.rewardCount + 2)):
                i = i % len(rewards)
                if rewards[i].x < self.pos[0] < rewards[i].x + rewards[i].width \
                        and rewards[i].y < self.pos[1] < rewards[i].y + rewards[i].height:
                    if not self.rewards[rewards[i]]:
                        self.reward_punishment += 1
                        self.rewardCount += 1
                        self.rewards[rewards[i]] = 1

    def update(self):
        """updates the state, whether it has reached the goal of left the borders of the game """
        _, distance = self.distanceTo(goal)
        if not self.dead and not self.reachedGoal:
            self.move()
            if self.pos[0] < 2 or self.pos[0] > 798 or self.pos[1] < 2 or self.pos[1] > 798:
                self.dead = True
            elif distance < 6:
                self.reachedGoal = True
                self.rewardCount += 1

    def calculateFitness(self):
        """determines the Fitness, how good of a path that it has taken. if it has reached the goal, it determines the
        fitness by inverse of its step count"""
        if self.rewardCount == len(self.rewards):
            _, distance = self.distanceTo(rewards[self.rewardCount - 1].midpoint)
            self.fitness = 10 + 1.0 / distance + self.reward_punishment
        elif self.reachedGoal:
            """taken 1.0/16.0 out, multiplied steps by 10, this should be enough too"""
            self.fitness = 10 + 100000.0 / (self.brain.step * self.brain.step) + self.reward_punishment
        else:
            _, distance = self.distanceTo(goal)
            self.fitness = 10 + 1.0 / distance + self.reward_punishment

    def mutate(self):
        """mutates the brain, duplicate function from brain because of OOP purposes"""
        self.brain.mutate()

    def giveBirth(self):
        """'gives birth' by cloning itself"""
        baby = Dot(self.brain.size, self.color1, self.color2, self.shape, self.mutationRate, self.decisionRate,
                   self.maxVel,beingBorn=True)
        baby.brain = self.brain
        baby.brain.step = 0
        return baby


class Population:
    """Population, group of Dots. Size is the amount of members. brainSize is the brain capacity of each member.
    color1 and color2 determines the color of the population,

    mutationRate determines how good they can clone their brain to the next generation,
    low mutationRate makes exact copy of the last generation, almost no possibility of exploration
    high mutationRate causes huge differences between generations, may lead to loss of paths that was found before

    decisionRate determines the intelligence of Population.
    low decisionRate makes them take almost the exact same path as ancestors, excluding the mutations
    high decisionRate makes them go almost directly to the next reward and finish the job as the best Population

     sheet allows it to save the Population to its own Excel sheet"""

    def __init__(self, size, brainSize, color1, color2, shape, mutationRate, decisionRate, maxVel):
        if load:
            self.sheet = loadExcel.sheet_by_index(len(populations))
            self.generation = int(substring_after(self.sheet.cell_value(0, 1), ": ")) + 1
            self.color1 = make_tuple(substring_after(self.sheet.cell_value(0, 2), ": "))
            self.color2 = make_tuple(substring_after(self.sheet.cell_value(0, 3), ": "))
            self.shape = substring_after(self.sheet.cell_value(0, 4), ": ")
            self.size = int(substring_after(self.sheet.cell_value(0, 5), ": "))
            self.brainSize = int(substring_after(self.sheet.cell_value(0, 6), ": "))
            self.mutationRate = float(substring_after(self.sheet.cell_value(0, 7), ": "))
            self.decisionRate = float(substring_after(self.sheet.cell_value(0, 8), ": "))
            self.maxVel = float(substring_after(self.sheet.cell_value(0, 9), ": "))
            self.screen = screen
            self.best = 0
            self.bestParents = []
            self.winners = 0
            self.dots = np.ndarray(self.size, dtype=object)
            self.nextDots = np.ndarray(self.size, dtype=object)
            self.maxVel = maxVel
            self.saveFile = open(loadFrom + "Population" + str(len(populations) + 1) + ".npy", "rb")
            self.loadbrain = np.load(self.saveFile,allow_pickle=True)
            for i in range(self.size):
                dot = Dot(brainSize, self.color1, self.color2, self.shape, self.mutationRate, self.decisionRate,
                          self.maxVel, self.loadbrain)
                self.dots[i] = dot
            self.saveFile = open(loadFrom + "Population" + str(len(populations) + 1) + ".npy", "wb")
            self.sheet = dataset.get_sheet(len(populations))
        else:
            self.brainSize = brainSize
            self.size = size
            self.dots = np.ndarray(self.size, dtype=object)
            self.nextDots = np.ndarray(self.size, dtype=object)
            self.screen = screen
            self.generation = 1
            self.best = 0
            self.bestParents = []
            self.winners = 0
            self.color1 = color1
            self.color2 = color2
            self.shape = shape
            self.mutationRate = mutationRate
            self.decisionRate = decisionRate
            self.maxVel = maxVel
            for i in range(self.size):
                dot = Dot(brainSize, self.color1, self.color2, self.shape, self.mutationRate, self.decisionRate,
                          self.maxVel)
                self.dots[i] = dot
            self.sheet = dataset.add_sheet("Population " + str(len(populations) + 1), cell_overwrite_ok=True)
            self.populateSheet()
            self.saveFileLocation = "./" + str(startDate) + "/Population" + str(len(populations) + 1) + ".npy"
            self.saveFile = open(self.saveFileLocation, "wb")

        print("all dots alive")
        populations.append(self)

    def populateSheet(self):
        """populates its own sheet with its characteristics, such as color, size, brainSize, mutationRate, decisionRate
        and the time it was created. + populates the name of each column"""
        self.sheet.write(0, 0, "Started at: " + str(datetime.datetime.now()))
        self.sheet.write(0, 2, "Color1: " + str(self.color1))
        self.sheet.write(0, 3, "Color2: " + str(self.color2))
        self.sheet.write(0, 4, "Shape: " + str(self.shape))
        self.sheet.write(0, 5, "Population Size: " + str(self.size))
        self.sheet.write(0, 6, "Brain Size: " + str(self.brainSize))
        self.sheet.write(0, 7, "Mutation Rate: " + str(self.mutationRate))
        self.sheet.write(0, 8, "Decision Rate: " + str(self.decisionRate))
        self.sheet.write(0, 9, "Maximum Velocity: " + str(self.maxVel))

        # Generation Stats
        self.sheet.write(1, 0, "Generations ")
        self.sheet.write(1, 1, "Best Dot's Position ")
        self.sheet.write(1, 2, "Best Dot's Fitness ")
        self.sheet.write(1, 3, "Best Dot's Score ")
        self.sheet.write(1, 4, "Step ")
        self.sheet.write(1, 5, "Winners ")

    def saveGeneration(self):
        """saves the generation statistics. the number, the best Dot, the best Dots fitness, score and step, and
        the amount of winners"""
        self.sheet.write(self.generation + 1, 0, str(self.generation))
        self.sheet.write(self.generation + 1, 1, str(self.bestParents[0]))
        self.sheet.write(self.generation + 1, 2, str(self.dots[self.bestParents[0]].fitness))
        self.sheet.write(self.generation + 1, 3, str(self.dots[self.bestParents[0]].reward_punishment))
        self.sheet.write(self.generation + 1, 4, str(self.dots[self.bestParents[0]].brain.step))
        self.sheet.write(self.generation + 1, 5, str(self.winners))
        self.sheet.write(0, 1, "Last Generation: " + str(self.generation))
        np.save(self.saveFile, self.dots[self.bestParents[0]].brain.directions)

    def show(self):
        """draws all the Dots"""
        for i in range(self.size):
            (self.dots[i]).draw(self.screen)

    def update(self):
        """updates all the Dots"""
        for i in range(self.size):
            (self.dots[i]).update()

    def allDead(self):
        """determines if all the Dots are dead. decision of ending the generation"""
        for i in range(self.size):
            if not self.dots[i].dead and not self.dots[i].reachedGoal:
                return False
        return True

    def calculateFitness(self):
        """calculates the fitness of each member"""
        for i in range(self.size):
            self.dots[i].calculateFitness()

    def selectParent(self):
        """selects the best 10 Dot by creating a hashMap of each Dot, with key as its fitness, value as its position in
        population.dots list, sorts the hashMap by its fitness and marks the best 10 Dots as bestParents and the best
         Dot as the best"""
        fitness = np.ndarray((2, self.size))
        self.bestParents = []
        for i in range(self.size):
            fitness[0, i] = self.dots[i].fitness
            fitness[1, i] = i
        fitness = fitness[:, fitness[0].argsort()]
        fitness = np.flip(fitness, axis=1)
        for i in range(int(self.size/5)):
            self.bestParents.append(int(fitness[1, i]))
        self.dots[self.bestParents[0]].isBest = True
        if self.dots[self.bestParents[0]].reachedGoal:
            print("Reached Goal in " + str(self.dots[self.bestParents[0]].brain.step) + " steps")
        self.winners = 0
        for i in range(self.size):
            if self.dots[i].reachedGoal:
                self.winners += 1

    def naturalSelection(self):
        """natural selection, creates the next generation as copies of the best 10 of the current generation
        the best one does not change its position in population.dots list.
        other babies are copies of randomly selected members from the best 10 list
        as last the next generation is copied over as the current generation and generation count is incremented"""
        print(colorList.get(self.color1) + "Starting Natural Selection")
        self.selectParent()
        time.sleep(0.001)
        self.saveGeneration()
        self.nextDots[self.best] = self.dots[self.best]

        for i in range(0, self.size):
            if i is self.best:
                i += 1
            rand = random.randint(0, int(self.size/5) - 1)
            Baby = self.dots[self.bestParents[rand]].giveBirth()
            self.nextDots[i] = Baby
            print(colorList.get(self.color1) + "Natural Selection For Population:" + str(colorList.get(self.color1))
                  + " " + str(self.shape) + " " + str(i))

        self.dots = self.nextDots
        self.generation += 1

    def mutate(self):
        """mutates all the dots' brains"""
        for i in range(self.size):
            self.dots[i].mutate()


def shapes(pos, color, shape):
    """draws the shape relative to the middle point of the shape"""
    shapeList = {
        "circle": 1,
        "triangle": 2,
        "square": 3,
        "ellipse": 4,
        "diamond": 5,
        "pentagon": 6,
        "hexagon": 7,
        "plus": 8
    }
    toDraw = shapeList.get(shape)
    if toDraw == 1:
        pygame.draw.circle(screen, color, pos, 5)
    elif toDraw == 2:
        pointA = np.array((pos[0], pos[1] - 6.64))
        pointB = np.array((pos[0] + 5.64, pos[1] + 3.32))
        pointC = np.array((pos[0] - 5.64, pos[1] + 3.32))
        pygame.draw.polygon(screen, color, (pointA, pointB, pointC))
    elif toDraw == 3:
        pointA = np.array((pos[0] - 2.5, pos[1] - 2.5))
        pointB = np.array((pos[0] + 2.5, pos[1] - 2.5))
        pointC = np.array((pos[0] + 2.5, pos[1] + 2.5))
        pointD = np.array((pos[0] - 2.5, pos[1] + 2.5))
        pygame.draw.polygon(screen, color, (pointA, pointB, pointC, pointD))
    elif toDraw == 4:
        pointA = np.array((pos[0] - 2.5, pos[1] - 1.25))
        rect = np.array((10, 5))
        pygame.draw.ellipse(screen, color, (pointA, rect))
    elif toDraw == 5:
        pointA = np.array((pos[0], pos[1] - 8.66))
        pointB = np.array((pos[0] + 5, pos[1]))
        pointC = np.array((pos[0], pos[1] + 8.66))
        pointD = np.array((pos[0] - 5, pos[1]))
        pygame.draw.polygon(screen, color, (pointA, pointB, pointC, pointD))
    elif toDraw == 6:
        pointA = np.array((pos[0], pos[1] - 5))
        pointB = np.array((pos[0] - 4.74, pos[1] - 1.54))
        pointC = np.array((pos[0] - 2.92, pos[1] + 4.04))
        pointD = np.array((pos[0] + 2.92, pos[1] + 4.04))
        pointE = np.array((pos[0] + 4.74, pos[1] - 1.54))
        pygame.draw.polygon(screen, color, (pointA, pointB, pointC, pointD, pointE))
    elif toDraw == 7:
        pointA = np.array((pos[0] - 2.5, pos[1] - 4.33))
        pointB = np.array((pos[0] + 2.5, pos[1] - 4.33))
        pointC = np.array((pos[0] + 5, pos[1]))
        pointD = np.array((pos[0] + 2.5, pos[1] + 4.33))
        pointE = np.array((pos[0] - 2.5, pos[1] + 4.33))
        pointF = np.array((pos[0] - 5, pos[1]))
        pygame.draw.polygon(screen, color, (pointA, pointB, pointC, pointD, pointE, pointF))
    elif toDraw == 8:
        pointA = np.array((pos[0] - 1, pos[1] - 5))
        pointB = np.array((pos[0] + 1, pos[1] - 5))
        pointC = np.array((pos[0] + 1, pos[1] - 1))
        pointD = np.array((pos[0] + 5, pos[1] - 1))
        pointE = np.array((pos[0] + 5, pos[1] + 1))
        pointF = np.array((pos[0] + 1, pos[1] + 1))
        pointG = np.array((pos[0] + 1, pos[1] + 5))
        pointH = np.array((pos[0] - 1, pos[1] + 5))
        pointI = np.array((pos[0] - 1, pos[1] + 1))
        pointJ = np.array((pos[0] - 5, pos[1] + 1))
        pointK = np.array((pos[0] - 5, pos[1] - 1))
        pointL = np.array((pos[0] - 1, pos[1] - 1))
        pygame.draw.polygon(screen, color, (pointA, pointB, pointC, pointD,
                                            pointE, pointF, pointG, pointH,
                                            pointI, pointJ, pointK, pointL))


def boxcollisions1():
    """iterates all populations' dots for collisions, gets handled by thread #2 """
    while not stopEvent2.is_set():
        for i in range(3):
            for dot in populations[i].dots:
                if not dot.dead:
                    dot.boxcollisions()


def boxcollisions2():
    """iterates all populations' dots for collisions, gets handled by thread #2 """
    while not stopEvent3.is_set():
        for i in range(3, 6):
            if not populations[i].allDead():
                for dot in populations[i].dots:
                    if not dot.dead:
                        dot.boxcollisions()


def rewardcollisions1():
    """iterates all populations' dots for collisions, gets handled by thread #2 """
    while not stopEvent4.is_set():
        for i in range(3):
            if not populations[i].allDead():
                for dot in populations[i].dots:
                    if not dot.dead:
                        dot.rewardcollisions()


def rewardcollisions2():
    """iterates all populations' dots for collisions, gets handled by thread #2 """
    while not stopEvent5.is_set():
        for i in range(3, 6):
            if not populations[i].allDead():
                for dot in populations[i].dots:
                    if not dot.dead:
                        dot.rewardcollisions()


def Printer(winners, generation, population):
    """prints the winners, generation count and population, helps OOP as creating a common function."""
    print(colorList.get(populations[population].color1) + "Population:" + str((population + 1)) + " {")
    print(colorList.get(populations[population].color1) + str(winners) + " winners in generation: " + str(generation))
    print(colorList.get(populations[population].color1) + "Started Generation " + str(generation + 1) + " }")


def populate(stopEvent, population):
    """starts and maintains a population. gets handled by a thread for simultaneous training"""
    while not stopEvent.is_set():
        if not populations[population].allDead():
            '''time.sleep(0.00001)'''
            populations[population].update()
            ''' pygame.display.update()'''
        else:
            populations[population].calculateFitness()
            # Printer(populations[population].winners, populations[population].generation, population)
            populations[population].naturalSelection()
            populations[population].mutate()


def loop():
    """gets handled by thread #1, first load all the other threads, then maintains all display stuff for all the
    objects. if pygame quits, stop all the threads, saves the workbook then quits."""
    if not testing:
        t2.start()
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()
        t9.start()
        t10.start()
        t11.start()
    while True:
        if not backgroundOnly:
            screen.fill(gray)
            for pop in populations:
                pop.show()
            for box in boxes:
                box.draw()
            for reward in rewards:
                reward.draw()
            pygame.draw.circle(screen, (255, 0, 0), goal, 5)
            pygame.display.update()
        schedule.run_pending()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                getPoint()
            if event.type == pygame.QUIT:
                time.sleep(0.5)
                stopEvent11.set()
                stopEvent10.set()
                stopEvent9.set()
                stopEvent8.set()
                stopEvent7.set()
                stopEvent6.set()
                stopEvent5.set()
                stopEvent4.set()
                stopEvent3.set()
                stopEvent2.set()
                stopEvent1.set()
                pygame.quit()
                if not testing:
                    saveFile()
                quit()


def saveFile():
    """save function for scheduler"""
    if not testing:
        try:
            dataset.save(datasetName)
        except:
            pass
        for pop in populations:
            if os.path.getsize(pop.saveFileLocation) > 10:
                shutil.copy(pop.saveFileLocation, pop.saveFileLocation + ".backup")


def getPoint():
    x, y = pygame.mouse.get_pos()
    print(x, y)


def substring_after(s, delim):
    return s.partition(delim)[2]


"""creating outer limit boxes"""
Box(0, 0, 10, 800)
Box(790, 0, 10, 800)
Box(0, 790, 800, 10)
Box(0, 0, 800, 10)

Box(375, 600, 5, 200)
Box(425, 650, 5, 150)
Box(375, 600, 200, 5)
Box(425, 650, 100, 5)
Box(525, 650, 5, 60)
Box(575, 600, 5, 150)
Box(475, 750, 105, 5)
Box(475, 700, 5, 50)
Box(630, 550, 5, 250)
Box(325, 550, 305, 5)
Box(320, 550, 5, 190)
Box(60, 740, 265, 5)
Box(10, 685, 255, 5)
Box(60, 620, 265, 5)
Box(10, 550, 255, 5)
Box(260, 445, 5, 105)
Box(260, 495, 450, 5)
Box(260, 445, 450, 55)
Box(710, 445, 5, 295)
Box(70, 395, 735, 5)
Box(130, 395, 5, 105)
Box(180, 445, 535, 5)
Box(60, 495, 155, 5)
Box(10, 445, 75, 5)
Box(10, 345, 335, 5)
Box(400, 300, 5, 100)
Box(345, 245, 5, 105)
Box(235, 245, 300, 5)
Box(455, 245, 5, 105)
Box(455, 345, 280, 5)
Box(735, 240, 5, 110)
Box(75, 190, 725, 5)
Box(585, 190, 5, 100)
Box(515, 290, 160, 5)
Box(645, 240, 95, 5)
Box(180, 190, 5, 100)
Box(75, 290, 205, 5)
Box(10, 245, 115, 5)
Box(10, 125, 700, 5)
Box(90, 65, 720, 5)

"""creating rewards"""
Reward(380, 650, 45, 5)
Reward(425, 605, 5, 45)
Reward(525, 605, 5, 45)
Reward(530, 650, 45, 5)
Reward(530, 705, 45, 5)
Reward(525, 710, 5, 40)
Reward(480, 700, 45, 5)
Reward(475, 655, 5, 45)
Reward(430, 700, 45, 5)
Reward(430, 750, 45, 5)
Reward(475, 755, 5, 35)
Reward(575, 755, 5, 35)
Reward(580, 750, 50, 5)
Reward(580, 600, 50, 5)
Reward(575, 555, 5, 45)
Reward(375, 555, 5, 45)
Reward(325, 600, 50, 5)
Reward(325, 740, 50, 5)
Reward(320, 745, 5, 45)
Reward(60, 745, 5, 45)
Reward(10, 740, 50, 5)
Reward(60, 690, 5, 50)
Reward(260, 690, 5, 50)
Reward(265, 685, 55, 5)
Reward(260, 625, 5, 60)
Reward(60, 625, 5, 60)
Reward(10, 620, 50, 5)
Reward(60, 555, 5, 65)
Reward(260, 555, 5, 65)
Reward(265, 550, 55, 5)
Reward(320, 500, 5, 50)
Reward(630, 500, 5, 50)
Reward(635, 550, 75, 5)
Reward(635, 735, 75, 5)
Reward(710, 740, 5, 50)
Reward(715, 735, 75, 5)
Reward(715, 445, 75, 5)
Reward(710, 400, 5, 45)
Reward(180, 400, 5, 45)
Reward(135, 445, 45, 5)
Reward(180, 450, 5, 45)
Reward(210, 450, 5, 45)
Reward(215, 495, 45, 5)
Reward(210, 500, 5, 50)
Reward(60, 500, 5, 50)
Reward(10, 495, 50, 5)
Reward(60, 450, 5, 45)
Reward(80, 450, 5, 45)
Reward(85, 445, 45, 5)
Reward(80, 400, 5, 45)
Reward(70, 400, 5, 45)
Reward(10, 395, 60, 5)
Reward(70, 350, 5, 45)
Reward(345, 350, 5, 45)
Reward(350, 345, 50, 5)
Reward(350, 300, 50, 5)
Reward(400, 250, 5, 50)
Reward(405, 300, 50, 5)
Reward(405, 345, 50, 5)
Reward(455, 350, 5, 45)
Reward(735, 350, 5, 45)
Reward(740, 345, 50, 5)
Reward(740, 240, 50, 5)
Reward(735, 195, 5, 45)
Reward(645, 195, 5, 45)
Reward(590, 240, 55, 5)
Reward(645, 245, 5, 45)
Reward(670, 245, 5, 45)
Reward(675, 290, 60, 5)
Reward(670, 295, 5, 50)
Reward(515, 295, 5, 50)
Reward(460, 290, 55, 5)
Reward(515, 250, 5, 40)
Reward(530, 250, 5, 40)
Reward(535, 245, 50, 5)
Reward(530, 195, 5, 50)
Reward(235, 195, 5, 50)
Reward(185, 245, 50, 5)
Reward(235, 250, 5, 40)
Reward(275, 250, 5, 40)
Reward(280, 290, 65, 5)
Reward(275, 295, 5, 50)
Reward(75, 295, 5, 50)
Reward(10, 290, 65, 5)
Reward(75, 250, 5, 40)
Reward(120, 250, 5, 40)
Reward(125, 245, 55, 5)
Reward(120, 195, 5, 50)
Reward(75, 195, 5, 50)
Reward(10, 190, 65, 5)
Reward(75, 130, 5, 60)
Reward(705, 130, 5, 60)
Reward(710, 125, 80, 5)
Reward(705, 70, 5, 55)
Reward(90, 70, 5, 55)
Reward(10, 65, 80, 5)
Reward(90, 10, 5, 55)
Reward(774, 34, 1, 1)
time.sleep(0.1)

'''
# This calculates the minimum steps required to reach the goal
total_distance = 0

for j in range(1,len(rewards)):

    Rdist1 = rewards[j].midpoint - rewards[j-1].midpoint
    Rdist2 = np.dot(Rdist1.T, Rdist1)
    Rdist3 = np.sqrt(Rdist2)
    total_distance += Rdist3

Rdist1 = goal - rewards[97].midpoint
Rdist2 = np.dot(Rdist1.T, Rdist1)
Rdist3 = np.sqrt(Rdist2)
total_distance += Rdist3


print(total_distance/5500)
exit()
'''

"""creating populations"""
pop1 = Population(10, 5000, red, pink, "circle", 0.01, 0.99, 1.85)
pop2 = Population(10, 5000, red, black, "diamond", 0.01, 0.99, 1.90)
pop3 = Population(10, 5000, red, magenta, "plus", 0.01, 0.99, 1.95)
pop4 = Population(10, 5000, blue, magenta, "diamond", 0.01, 0.99, 2)
pop5 = Population(10, 5000, cyan, navy, "pentagon", 0.01, 0.99, 2.05)
pop6 = Population(10, 5000, yellow, green, "hexagon", 0.01, 0.99, 2.5)

"""main function to start everything"""
if __name__ == '__main__':
    if not testing:
        schedule.every(1).minutes.do(saveFile)
    t1 = Thread(target=loop)
    t2 = Thread(target=boxcollisions1)
    t3 = Thread(target=boxcollisions2)
    t4 = Thread(target=rewardcollisions1)
    t5 = Thread(target=rewardcollisions2)
    t6 = Thread(target=populate, args=(stopEvent6, 0))
    t7 = Thread(target=populate, args=(stopEvent7, 1))
    t8 = Thread(target=populate, args=(stopEvent8, 2))
    t9 = Thread(target=populate, args=(stopEvent9, 3))
    t10 = Thread(target=populate, args=(stopEvent10, 4))
    t11 = Thread(target=populate, args=(stopEvent11, 5))
    t1.start()
