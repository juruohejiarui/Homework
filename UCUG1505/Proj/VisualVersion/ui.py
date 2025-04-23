import detect_face
import time
import math
import numpy as np
import pygame as pyg
import lib

screenWidth = 360
screenHeight = 640

PlayerRadius = 30
BallRadius = 20
DefenderRadius = 25

PlayerMass = 30
BallMass = 10
DefenderMass = 25

GateHeight = BallRadius

DefenderPosY = GateHeight + 10 + DefenderRadius

radius = [PlayerRadius, BallRadius, DefenderRadius]
mass = [PlayerMass, BallMass, DefenderMass]

def calcPlayerPos(curPlayerPos : np.ndarray, nxtPlayerPos : np.ndarray) :
	dis : np.ndarray = np.sqrt(((nxtPlayerPos - curPlayerPos) ** 2).sum()) + 1
	return (curPlayerPos + (nxtPlayerPos - curPlayerPos) * min(max(0.1, math.log(dis.item()) / 5), 0.5))

def calcPlayerVelocity(curPlayerPos : np.ndarray, nxtPlayerPos : np.ndarray, deltaTime : float) -> np.ndarray :
	return (nxtPlayerPos - curPlayerPos) / deltaTime

def checkInDefendRange(curBallPos : np.ndarray) -> bool :
	if curBallPos[1] < screenHeight / 3 : 
		return True
	return False

def calcDefenderTarget(curPlayerPos : np.ndarray, curBallPos : np.ndarray, curBallVel : np.ndarray) -> np.ndarray :
	intersect1 = lib.intersection(np.array([0, DefenderPosY]), np.array([screenWidth, DefenderPosY]), curBallPos, curBallPos + curBallVel)
	intersect2 = lib.intersection(np.array([0, DefenderPosY]), np.array([screenWidth, DefenderPosY]), curPlayerPos, curBallPos)
	realintersect = None
	if intersect1 is None and intersect2 is None :
		realintersect = curBallPos
	elif intersect1 is None :
		realintersect = intersect2
	elif intersect2 is None :
		realintersect = intersect1
	else :
		realintersect = (intersect1 + intersect2) / 2
	# constraint the defender to the screen width
	realintersect[0] = max(DefenderRadius, min(realintersect[0], screenWidth - DefenderRadius))
	return realintersect
	

def calcCofPlayer(velocity : np.ndarray) -> np.ndarray :
	return max(np.linalg.norm(velocity) / 50, 10) / 20 + 0.1

def checkInGate(pos : np.ndarray) -> bool :
	if 40 < pos[0] < screenWidth - 40 and pos[1] < GateHeight + 5 :
		return True
	return False


def calcCollision(pos : list[np.ndarray], velocity : list[np.ndarray], mass : list[float], radius : list[float], C : list[float], deltaTime) -> tuple[list[np.ndarray], list[np.ndarray]] :
	newPos : list[np.ndarray] = []
	newVelocity : list[np.ndarray] = []
	# print(pos, velocity)
	for i in range(len(pos)) :
		newPos.append(pos[i] + velocity[i] * deltaTime)
		newVelocity.append(velocity[i])
	for i in range(len(pos)) :
		for j in range(i + 1, len(pos)) :
			dis = np.sqrt(((newPos[i] - newPos[j]) ** 2).sum()).item()
			
			if dis < radius[i] + radius[j] :
				# apply momentum conservation
				deltaPos = newPos[i] - newPos[j]
				deltaVel = newVelocity[i] - newVelocity[j]

				newPos[i] += deltaPos * (radius[i] + radius[j] - dis + 2) / (2 * dis)
				newPos[j] -= deltaPos * (radius[i] + radius[j] - dis + 2) / (2 * dis)

				t = newVelocity[i] * mass[i] + newVelocity[j] * mass[j]
				# for velocity
				newVelocity[i] = (-deltaVel * mass[j] * min(C[j], C[i]) + t) / (mass[i] + mass[j])
				newVelocity[j] = (deltaVel * mass[i] * min(C[j], C[i]) + t) / (mass[i] + mass[j])
	return newPos, newVelocity

def calcCollisionWithWall(posLst : list[np.ndarray], velocity : list[np.ndarray], mass : list[float], radius : list[float], deltaTime) -> tuple[list[np.ndarray], list[np.ndarray]] :
	newPos, newVelocity = [], []
	for i in range(len(posLst)) :
		newPos.append(posLst[i])
		newVelocity.append(velocity[i])
	for i in range(len(posLst)) :
		if newPos[i][0] < radius[i] or newPos[i][0] > screenWidth - radius[i] :
			newVelocity[i] = np.array([-newVelocity[i][0], newVelocity[i][1]])
			newPos[i] = (max(radius[i], min(newPos[i][0], screenWidth - radius[i])), newPos[i][1])
		if newPos[i][1] < radius[i] or newPos[i][1] > screenHeight - radius[i] :
			newVelocity[i] = np.array([newVelocity[i][0], -newVelocity[i][1]])
			newPos[i] = (newPos[i][0], max(radius[i], min(newPos[i][1], screenHeight - radius[i])))
	return newPos, newVelocity

def main() :
	pyg.init()
	screen = pyg.display.set_mode((screenWidth, screenHeight))
	
	done, needReflesh = False, True

	curTime = time.monotonic()

	inDefendRangeStartTime = 0
	inDefendRangeLstState = False

	font = pyg.font.Font("./CascadiaCodeNF.ttf", 20)

	while not done :
		if needReflesh :
			curPlayerPos, targetPlayerPos = np.array([screenWidth / 2, screenHeight / 2]), np.array([screenWidth / 2, screenHeight / 2])
			curDefenderPos = np.array([screenWidth / 2, ])
			curBallPos, curBallVel = np.array([screenWidth / 2, screenHeight / 2]), np.zeros(2)
			startTime = time.monotonic()
			gameEnd = False
			needReflesh = False

		if not gameEnd :
			head = [detect_face.detect()]
			if head[0] :
				targetPlayerPos = np.array(
					[(-head[0][0] + 0.5) * screenWidth * 2 + screenWidth / 2, 
					head[0][1] * screenHeight + screenHeight / 8])
				targetPlayerPos[0] = max(PlayerRadius, min(targetPlayerPos[0], screenWidth - PlayerRadius))
				targetPlayerPos[1] = max(PlayerRadius, min(targetPlayerPos[1], screenHeight - PlayerRadius))
			
			deltaTime = time.monotonic() - curTime
			curTime = time.monotonic()
			newPlayerPos = calcPlayerPos(curPlayerPos, targetPlayerPos)
			newDefenderPos = calcPlayerPos(curDefenderPos, calcDefenderTarget(newPlayerPos, curBallPos, curBallVel))
			playerVel = calcPlayerVelocity(curPlayerPos, newPlayerPos, deltaTime)
			defenderVel = calcPlayerVelocity(curDefenderPos, newDefenderPos, deltaTime)

			curPos, curVel = calcCollision([curPlayerPos, curBallPos, curDefenderPos], 
								  	[playerVel, curBallVel, defenderVel],
									mass, radius, 
									[calcCofPlayer(playerVel), 1, 0.8], 
									deltaTime)
			curPlayerPos, curDefenderPos = newPlayerPos, newDefenderPos
			curPos, curVel = calcCollisionWithWall([curPos[1]], [curVel[1]], [BallMass], [BallRadius], deltaTime)
			curBallPos = curPos[0]
			curBallVel = curVel[0]

			inDefendRangeNewState = checkInDefendRange(curBallPos)
			if inDefendRangeNewState :
				if not inDefendRangeLstState :
					inDefendRangeStartTime = time.monotonic()

			inDefendRangeLstState = inDefendRangeNewState

			# calculate speed using 
			# show the Player on the screen
			screen.fill((255, 255, 255))
			# gate
			pyg.draw.rect(screen, (0, 255, 0), (40, 0, screenWidth - 40 * 2, GateHeight))
			# player
			pyg.draw.circle(screen, (255, 0, 0), (int(curPlayerPos[0]), int(curPlayerPos[1])), PlayerRadius)
			# ball
			pyg.draw.circle(screen, (0, 0, 255), (int(curBallPos[0]), int(curBallPos[1])), BallRadius)
			# defender
			pyg.draw.circle(screen, (0, 255, 255), (int(curDefenderPos[0]), int(curDefenderPos[1])), DefenderRadius)
			# show remaining time
			remainTime = 30 - (time.monotonic() - startTime)
			if remainTime < 0 :
				remainTime = 0
			text = font.render("Time: " + str(int(remainTime)), True, (0, 0, 0))
			screen.blit(text, (screenWidth / 2 - text.get_width() / 2, screenHeight - text.get_height() - 10))

			if inDefendRangeLstState :
				remainTime = 10 - (time.monotonic() - inDefendRangeStartTime)
				if remainTime < 0 :
					remainTime = 0
				text = font.render(str(int(remainTime)), True, (0, 0, 0))
				screen.blit(text, (screenWidth / 2 - text.get_width() / 2, screenHeight - text.get_height() * 2 - 10))


		if checkInGate(curBallPos) :
			text = font.render("Goal!", True, (0, 0, 0))
			screen.blit(text, (screenWidth / 2 - text.get_width() / 2, screenHeight / 2 - text.get_height() / 2))
			gameEnd = True
			pyg.display.update()
		if time.monotonic() - startTime > 30 or (inDefendRangeLstState and time.monotonic() - inDefendRangeStartTime > 10) :
			text = font.render("Time Out!", True, (255, 0, 0))
			screen.blit(text, (screenWidth / 2 - text.get_width() / 2, screenHeight / 2 - text.get_height() / 2))
			gameEnd = True
			pyg.display.update()
		

		# receive key, if press 'r', restart the game
		keys = pyg.key.get_pressed()
		if keys[pyg.K_r] :
			needReflesh = True
		if keys[pyg.K_ESCAPE] :
			done = True
		for event in pyg.event.get() :
			if event.type == pyg.QUIT :
				done = True
		if not gameEnd :
			pyg.display.update()

if __name__ == "__main__" :
	main()