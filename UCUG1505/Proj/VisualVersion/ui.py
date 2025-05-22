import detect_face
import time
import math
import numpy as np
import pygame as pyg
import lib

screenWidth = 400
screenHeight = 800

PlayerRadius = 30
BallRadius = 10
DefenderRadius = 15

PlayerMass = 30
BallMass = 10
DefenderMass = 30

BorderWidth = 25

GateRadius = 70
Winner = 0

ScorePlayer, ScoreDefender = 0, 0

BallInitPos = np.array([screenWidth / 2, screenHeight / 2])
PlayerInitPos = np.array([screenWidth / 2, screenHeight * 3 / 4])
GatePos = np.array([screenWidth / 2, 10])
OwnGatePos = np.array([screenWidth / 2, screenHeight - 10])

DefenderPosY = GateRadius + 10 + DefenderRadius

radius = [PlayerRadius, BallRadius, DefenderRadius]
mass = [PlayerMass, BallMass, DefenderMass]

def calcPlayerPos(curPlayerPos : np.ndarray, nxtPlayerPos : np.ndarray) :
	dis : np.ndarray = np.sqrt(((nxtPlayerPos - curPlayerPos) ** 2).sum()) + 1
	vel = (nxtPlayerPos - curPlayerPos) * min(max(0.1, math.log(dis.item()) / 5), 0.4)
	return (curPlayerPos + vel)

def calcPlayerVelocity(curPlayerPos : np.ndarray, nxtPlayerPos : np.ndarray, deltaTime : float) -> np.ndarray :
	return (nxtPlayerPos - curPlayerPos) / deltaTime

def calcDefenderTarget(curBallPos : np.ndarray) -> np.ndarray :
	vec = curBallPos - GatePos
	return vec * (GateRadius + DefenderRadius + 20) / np.linalg.norm(vec) + GatePos

def calcDefenderPos(curDefenderPos : np.ndarray, nxtDefenderPos : np.ndarray) :
	dis : np.ndarray = np.sqrt(((nxtDefenderPos - curDefenderPos) ** 2).sum()) + 1
	return (curDefenderPos + (nxtDefenderPos - curDefenderPos) * min(max(0.1, math.log(dis.item()) / 10), 0.2))

def calcDefenderNxtPos(curDefenderPos : np.ndarray, curBallPos : np.ndarray, curBallVel : np.ndarray, deltaTime) -> np.ndarray :
	ballPos = curBallPos + curBallVel * deltaTime / 2
	if np.linalg.norm(ballPos - curDefenderPos) < 60 and np.linalg.norm(ballPos - GatePos) < GateRadius + DefenderRadius + 160 :
		dis = np.linalg.norm(ballPos - OwnGatePos)
		if np.linalg.norm(curBallVel) < 30:
			trg = curDefenderPos + (ballPos - curDefenderPos)
		else :
			trg = OwnGatePos + (ballPos - OwnGatePos) * (dis + DefenderRadius + BallRadius / 2) / dis
		trg = curDefenderPos + (trg - curDefenderPos) * (0.4 + np.linalg.norm(trg - curDefenderPos) / 60 * 0.2)
	else :
		vec = ballPos - GatePos
		trg = vec * (GateRadius + DefenderRadius + 20) / np.linalg.norm(vec) + GatePos
		trg = calcDefenderPos(curDefenderPos, trg)
	return trg

def calcCofPlayer(velocity : np.ndarray) -> float :
	return max(np.linalg.norm(velocity) / 50, 10) / 20 + 0.1

def checkInGate(pos : np.ndarray) -> np.bool_ :
	vec = pos - GatePos
	return np.linalg.norm(vec) < GateRadius

def checkInOwnGate(pos : np.ndarray) -> np.bool_ :
	vec = pos - OwnGatePos
	return np.linalg.norm(vec) < GateRadius

def drawbackground(screen) :
	# 颜色设置
	ice_color = (180, 220, 255)
	line_color = (255, 0, 0)
	line_width = 4

	# 创建桌面背景
	screen.fill(ice_color)

	# 中线（横着的）
	pyg.draw.line(screen, line_color, (25, screenHeight/2), (screenWidth-25, screenHeight/2), line_width)

	# 发球圈
	pyg.draw.circle(screen, line_color, (screenWidth/2,screenHeight/2), 50, line_width)

	# 顶部球门圈
	pyg.draw.circle(screen, (200,0,0), (GatePos[0], GatePos[1]), GateRadius, line_width)
	pyg.draw.circle(screen, (255,20,147,150), (GatePos[0], GatePos[1]), GateRadius-line_width+1)

	# 底部球门圈
	pyg.draw.circle(screen, (200,0,0), (screenWidth/2, screenHeight-10), GateRadius, line_width)
	pyg.draw.circle(screen, (255,20,147,150), (screenWidth/2, screenHeight-10), GateRadius-line_width+1)

	# 边界矩形
	pyg.draw.rect(screen, (0,191,255,100), pyg.Rect(0, 0, screenWidth, screenHeight), 25)
	
	# 高光模拟（竖直椭圆形）
	highlight = pyg.Surface((400, 800), pyg.SRCALPHA)
	pyg.draw.ellipse(highlight, (255, 255, 255, 40), (-100, screenHeight*4/6, screenWidth+200, 400))
	screen.blit(highlight, (0, 0))
 
def drawplayer(screen, position) :
	pyg.draw.circle(screen, (160,0,30), position, 28)
	pyg.draw.circle(screen, (220,20,60), position, 24)
	pyg.draw.circle(screen, (0,0,0), position, 10)
	pyg.draw.circle(screen, (160,0,30), (position[0], position[1]-0.6), 8)
	pyg.draw.circle(screen, (220,20,60), (position[0],position[1]-1.2), 6)

def drawball(screen, position) :
	pyg.draw.circle(screen, (0,0,125), position, BallRadius)
	pyg.draw.circle(screen, (0,0,255), position, BallRadius-3)

def drawdefender(screen, position) :
	pyg.draw.circle(screen, (0, 100, 0), position, 20)              
	pyg.draw.circle(screen, (0, 180, 0), position, 17)              
	pyg.draw.circle(screen, (0, 0, 0), position, 7)               
	pyg.draw.circle(screen, (0, 100, 0), (position[0], position[1]-1), 6) 
	pyg.draw.circle(screen, (0, 180, 0), (position[0], position[1]-2), 4)  

def calcCollision(pos : list[np.ndarray], velocity : list[np.ndarray], mass : list[float], radius : list[float], C : list[float], deltaTime) -> tuple[list[np.ndarray], list[np.ndarray]] :
	newPos : list[np.ndarray] = []
	newVelocity : list[np.ndarray] = []
	for i in range(len(pos)) :
		# if (np.linalg.norm(vel) > 200) : vel *= 200 / np.linalg.norm(vel)
		newPos.append(pos[i] + velocity[i] * deltaTime)
		newVelocity.append(velocity[i])
	for i in range(len(pos)) :
		for j in range(i + 1, len(pos)) :
			dis = np.sqrt(((newPos[i] - newPos[j]) ** 2).sum()).item()
			
			if dis < radius[i] + radius[j] :
				# apply momentum conservation
				deltaPos = newPos[i] - newPos[j]
				deltaVel = newVelocity[i] - newVelocity[j]

				newPos[i] += deltaPos * (radius[i] + radius[j] - dis + 2) * mass[i] / ((mass[i] + mass[j]) * dis + 1e-5)
				newPos[j] -= deltaPos * (radius[i] + radius[j] - dis + 2) * mass[j] / ((mass[i] + mass[j]) * dis + 1e-5)

				t = newVelocity[i] * mass[i] + newVelocity[j] * mass[j]
				# for velocity
				newVelocity[i] = (-deltaVel * mass[j] * min(C[j], C[i]) + t) / (mass[i] + mass[j])
				newVelocity[j] = (deltaVel * mass[i] * min(C[j], C[i]) + t) / (mass[i] + mass[j])
 
	for i in range(len(newVelocity)) : 
		newVelocity[i] = newVelocity[i] * (1 - 0.5 * deltaTime)
	

	return newPos, newVelocity

def calcCollisionWithWall(posLst : list[np.ndarray], velocity : list[np.ndarray], C : list[float], radius : list[float], deltaTime) -> tuple[list[np.ndarray], list[np.ndarray]] :
	newPos, newVelocity = [], []
	for i in range(len(posLst)) :
		newPos.append(posLst[i])
		newVelocity.append(velocity[i])
	for i in range(len(posLst)) :
		if newPos[i][0] < radius[i] + BorderWidth or newPos[i][0] > screenWidth - radius[i] - BorderWidth :
			newVelocity[i] = np.array([C[i] * -newVelocity[i][0], newVelocity[i][1]])
			newPos[i] = np.array([max(radius[i] + BorderWidth, min(newPos[i][0], screenWidth - radius[i] - BorderWidth)), newPos[i][1]])
		if newPos[i][1] < radius[i] + BorderWidth or newPos[i][1] > screenHeight - radius[i] - BorderWidth :
			newVelocity[i] = np.array([newVelocity[i][0], C[i] * -newVelocity[i][1]])
			newPos[i] = np.array([newPos[i][0], max(radius[i] + BorderWidth, min(newPos[i][1], screenHeight - radius[i] - BorderWidth))])
	return newPos, newVelocity

def main() :
	global ScorePlayer, ScoreDefender
	pyg.init()
	screen = pyg.display.set_mode((screenWidth, screenHeight))
	
	done, needReflesh = False, True

	curTime = time.monotonic()


	font = pyg.font.Font("./CascadiaCodeNF.ttf", 20)

	while not done :
		if needReflesh :
			curPlayerPos, targetPlayerPos = PlayerInitPos.copy(), PlayerInitPos.copy()
			curDefenderPos = np.array([screenWidth / 2, screenHeight / 3])
			curBallPos, curBallVel = BallInitPos.copy(), np.zeros(2)
			startTime = time.monotonic()
			gameEnd = False
			needReflesh = False

		deltaTime = time.monotonic() - curTime
		curTime = time.monotonic()
		if not gameEnd :
			head = [detect_face.detect()]
			if head[0] :
				targetPlayerPos = np.array(
					[(-head[0][0] + 0.5) * screenWidth * 4 + screenWidth / 2, 
					(head[0][1] - 0.5) * screenHeight * 2 + screenHeight * 2 / 3])
				targetPlayerPos[0] = max(PlayerRadius, min(targetPlayerPos[0], screenWidth - PlayerRadius))
				targetPlayerPos[1] = max(PlayerRadius, min(targetPlayerPos[1], screenHeight - PlayerRadius))
			
			
			newPlayerPos = calcPlayerPos(curPlayerPos, targetPlayerPos)
			newDefenderPos = calcDefenderNxtPos(curDefenderPos, curBallPos, curBallVel, deltaTime)
			playerVel = calcPlayerVelocity(curPlayerPos, newPlayerPos, deltaTime)
			defenderVel = calcPlayerVelocity(curDefenderPos, newDefenderPos, deltaTime)

			curPos, curVel = calcCollision([curPlayerPos, curBallPos, curDefenderPos], 
								  	[playerVel, curBallVel, defenderVel],
									mass, radius, 
									[calcCofPlayer(playerVel), 0.9, 1], 
									deltaTime)
			curPlayerPos, curDefenderPos = newPlayerPos, newDefenderPos
			curPos, curVel = calcCollisionWithWall([curPos[1]], [curVel[1]], [1], [BallRadius], deltaTime)
			curBallPos = curPos[0]
			curBallVel = curVel[0]
		else :
			# only use collision
			curPos, curVel = calcCollision([curPlayerPos, curBallPos, curDefenderPos],
								  	[playerVel, curBallVel, defenderVel],
									mass, radius, 
									[calcCofPlayer(playerVel), 0.9, 1], 
									deltaTime)
			curPos, curVel = calcCollisionWithWall(curPos, curVel, [calcCofPlayer(playerVel), 1, 1], radius, deltaTime)
			curPlayerPos, curBallPos, curDefenderPos = curPos[0], curPos[1], curPos[2]
			playerVel, curBallVel, defenderVel = curVel[0], curVel[1], curVel[2]

		# calculate speed using 
		# show the Player on the screen
		drawbackground(screen)
		# gate
		# pyg.draw.rect(screen, (0, 255, 0), (40, 0, screenWidth - 40 * 2, GateHeight))
		# player
		drawplayer(screen, curPlayerPos)

		#pyg.draw.circle(screen, (255, 0, 0), (int(curPlayerPos[0]), int(curPlayerPos[1])), PlayerRadius)
		# ball
		drawball(screen, curBallPos)
		# defender
		drawdefender(screen, curDefenderPos)

		# draw score 
		# draw defender score
		text = font.render(str(ScoreDefender), True, (0, 0, 0), (255, 255, 255))
		pyg.draw.circle(screen, (255, 255, 255), (0, 0), 30 * math.sqrt(2))
		screen.blit(text, (5, 0))

		# draw player score
		text = font.render(str(ScorePlayer), True, (0, 0, 0), (255, 255, 255))
		pyg.draw.circle(screen, (255, 255, 255), (screenWidth, screenHeight), 30 * math.sqrt(2))
		screen.blit(text, (screenWidth - text.get_width() - 5, screenHeight - text.get_height() - 5))

		if gameEnd :
			if Winner == 1 :
				text = font.render("You Win!", True, (0, 0, 0), (255, 255, 255))
			elif Winner == 2 :
				text = font.render("You Lose!", True, (0, 0, 0), (255, 255, 255))
			screen.blit(text, (screenWidth / 2 - text.get_width() / 2, screenHeight / 2 - text.get_height() / 2))

		if not gameEnd :
			# check if the ball is in the gate
			if checkInGate(curBallPos) :
				Winner = 1
				if not gameEnd :
					ScorePlayer += 1
					gameEnd = True
			elif checkInOwnGate(curBallPos) :
				Winner = 2
				if not gameEnd :
					ScoreDefender += 1
					gameEnd = True
			else : Winner = 0
				
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