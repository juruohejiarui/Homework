import detect
import time
import math
import numpy as np
import pygame as pyg

screenWidth = 480
screenHeight = 1280

StickerRadius = 50
BallRadius = 30

StickerMass = 100
BallMass = 10

radius = [StickerRadius, BallRadius]
mass = [StickerMass, BallMass]

def calcStickerPos(curStickerPos, nxtStickerPos) :
	dis = math.sqrt((nxtStickerPos[0] - curStickerPos[0]) ** 2 + (nxtStickerPos[1] - curStickerPos[1]) ** 2)
	return (curStickerPos[0] + (nxtStickerPos[0] - curStickerPos[0]) * min(max(0.1, math.log(dis + 1) / 5), 0.5), 
			curStickerPos[1] + (nxtStickerPos[1] - curStickerPos[1]) * min(max(0.1, math.log(dis + 1) / 5), 0.5))

def calcSitckerVelocity(curStickerPos, nxtStickerPos, deltaTime) -> tuple[float, float]:
	return ((nxtStickerPos[0] - curStickerPos[0]) / deltaTime,
			(nxtStickerPos[1] - curStickerPos[1]) / deltaTime)

def calcCollision(pos : list[tuple[float, float]], velocity : list[tuple[float, float]], mass : list[float], radius : list[float], C : list[float], deltaTime) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] :
	# posLst : list of positions
	# velocity : list of velocity
	# mass : list of mass
	# radius : list of radius
	# deltaTime : time interval
	newPos = []
	newVelocity = []
	# print(pos, velocity)
	for i in range(len(pos)) :
		newPos.append((pos[i][0] + velocity[i][0] * deltaTime, pos[i][1] + velocity[i][1] * deltaTime))
		newVelocity.append((velocity[i][0], velocity[i][1]))
	for i in range(len(pos)) :
		for j in range(i + 1, len(pos)) :
			dis = math.sqrt((newPos[i][0] - newPos[j][0]) ** 2 + (newPos[i][1] - newPos[j][1]) ** 2)
			
			if dis < radius[i] + radius[j] :
				# apply momentum conservation
				# for position
				# stand away from each other
				delX = newPos[i][0] - newPos[j][0]
				delY = newPos[i][1] - newPos[j][1]
				delVX = newVelocity[i][0] - newVelocity[j][0]
				delVY = newVelocity[i][1] - newVelocity[j][1]

				newPos[i] = (newPos[i][0] + delX / dis * 1.01 * (radius[i] + radius[j] - dis) / 2, newPos[i][1] + delY / dis * 1.01 * (radius[i] + radius[j] - dis) / 2)
				newPos[j] = (newPos[j][0] - delX / dis * 1.01 * (radius[i] + radius[j] - dis) / 2, newPos[j][1] - delY / dis * 1.01 * (radius[i] + radius[j] - dis) / 2)
				# for velocity
				# v1 = (v1 * m1 + v2 * m2) / (m1 + m2)
				# v2 = (v1 * m1 + v2 * m2) / (m1 + m2)
				newVelocity[i] = ((delVX * C[i] + newVelocity[i][0] * mass[i] + newVelocity[j][0] * mass[j]) / (mass[i] + mass[j]),
								(delVY * C[i] + newVelocity[i][1] * mass[i] + newVelocity[j][1] * mass[j]) / (mass[i] + mass[j]))
				newVelocity[j] = ((-delVX * C[i] + newVelocity[i][0] * mass[i] + newVelocity[j][0] * mass[j]) / (mass[i] + mass[j]),
								(-delVY * C[i] + newVelocity[i][1] * mass[i] + newVelocity[j][1] * mass[j]) / (mass[i] + mass[j]))
	return newPos, newVelocity

def calcCollisionWithWall(posLst : list[tuple[float, float]], velocity : list[tuple[float, float]], mass : list[float], radius : list[float], deltaTime) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] :
	# posLst : list of positions
	# velocity : list of velocity
	# mass : list of mass
	# radius : list of radius
	# deltaTime : time interval
	newPos, newVelocity = [], []
	for i in range(len(posLst)) :
		newPos.append((posLst[i][0], posLst[i][1]))
		newVelocity.append((velocity[i][0], velocity[i][1]))
	for i in range(len(posLst)) :
		if newPos[i][0] < radius[i] or newPos[i][0] > screenWidth - radius[i] :
			newVelocity[i] = (-newVelocity[i][0], newVelocity[i][1])
			newPos[i] = (max(radius[i], min(newPos[i][0], screenWidth - radius[i])), newPos[i][1])
		if newPos[i][1] < radius[i] or newPos[i][1] > screenHeight - radius[i] :
			newVelocity[i] = (newVelocity[i][0], -newVelocity[i][1])
			newPos[i] = (newPos[i][0], max(radius[i], min(newPos[i][1], screenHeight - radius[i])))
	return newPos, newVelocity

def main() :
	pyg.init()
	screen = pyg.display.set_mode((screenWidth, screenHeight))
	(hands, faces) = detect.detect()
	curStickerPos, targetStickerPos = (screenWidth / 2, screenHeight / 2), (screenWidth / 2, screenHeight / 2)
	curBallPos, curBallVelocity = (screenWidth / 2, screenHeight / 2), (0, 0)
	done = False

	curTime = time.monotonic()
	
	while not done :
		(hands, faces) = detect.detect()
		if hands :
			targetStickerPos = ((0.5 - hands[0][0]) * screenWidth * 2 + screenWidth / 2, -(1 - hands[0][1]) * screenHeight / 2 + screenHeight / 2)
		
		deltaTime = time.monotonic() - curTime
		curTime = time.monotonic()
		newStickerPos = calcStickerPos(curStickerPos, targetStickerPos)
		stickerVelocity = calcSitckerVelocity(curStickerPos, newStickerPos, deltaTime)

		curPos, curVel = calcCollision([curStickerPos, curBallPos], [stickerVelocity, curBallVelocity], mass, radius, [0, 0], deltaTime)
		curStickerPos = curPos[0]
		curPos, curVel = calcCollisionWithWall([curPos[1]], [curVel[1]], [BallMass], [BallRadius], deltaTime)
		curBallPos = curPos[0]
		curBallVelocity = curVel[0]

		# calculate speed using 
		# show the Sticker on the screen
		screen.fill((255, 255, 255))
		pyg.draw.circle(screen, (255, 0, 0), (int(curStickerPos[0]), int(curStickerPos[1])), StickerRadius)
		pyg.draw.circle(screen, (0, 0, 255), (int(curBallPos[0]), int(curBallPos[1])), BallRadius)
		for event in pyg.event.get() :
			if event.type == pyg.QUIT :
				done = True
		pyg.display.update()
if __name__ == "__main__" :
	main()