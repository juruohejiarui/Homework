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
DefenderMass = 25

BorderWidth = 25

GateRadius = 70

BallInitPos = np.array([screenWidth / 2, screenHeight * 2 / 3])
PlayerInitPos = np.array([screenWidth / 2, screenHeight * 3 / 4])
GatePos = np.array([screenWidth / 2, 10])

DefenderPosY = GateRadius + 10 + DefenderRadius

radius = [PlayerRadius, BallRadius, DefenderRadius]
mass = [PlayerMass, BallMass, DefenderMass]

def calcPlayerPos(curPlayerPos : np.ndarray, nxtPlayerPos : np.ndarray) :
	dis : np.ndarray = np.sqrt(((nxtPlayerPos - curPlayerPos) ** 2).sum()) + 1
	vel = (nxtPlayerPos - curPlayerPos) * min(max(0.1, math.log(dis.item()) / 5), 0.5)
	return (curPlayerPos + vel)

def calcPlayerVelocity(curPlayerPos : np.ndarray, nxtPlayerPos : np.ndarray, deltaTime : float) -> np.ndarray :
	return (nxtPlayerPos - curPlayerPos) / deltaTime

def checkInDefendRange(curBallPos : np.ndarray) -> bool :
	if curBallPos[1] < screenHeight / 2 : 
		return True
	return False

def calcDefenderTarget(curBallPos : np.ndarray) -> np.ndarray :
	vec = curBallPos - GatePos
	return vec * (GateRadius + DefenderRadius + 20) / np.linalg.norm(vec) + GatePos

def calcDefenderPos(curDefenderPos : np.ndarray, nxtDefenderPos : np.ndarray) :
	dis : np.ndarray = np.sqrt(((nxtDefenderPos - curDefenderPos) ** 2).sum()) + 1
	return (curDefenderPos + (nxtDefenderPos - curDefenderPos) * min(max(0.05, math.log(dis.item()) / 10), 0.1))
	

def calcCofPlayer(velocity : np.ndarray) -> np.ndarray :
	return max(np.linalg.norm(velocity) / 50, 10) / 20 + 0.1

def checkInGate(pos : np.ndarray) -> bool :
	vec = pos - GatePos
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
	# print(pos, velocity)
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

				newPos[i] += deltaPos * (radius[i] + radius[j] - dis + 2) / (2 * dis + 1e-5)
				newPos[j] -= deltaPos * (radius[i] + radius[j] - dis + 2) / (2 * dis + 1e-5)

				t = newVelocity[i] * mass[i] + newVelocity[j] * mass[j]
				# for velocity
				newVelocity[i] = (-deltaVel * mass[j] * min(C[j], C[i]) + t) / (mass[i] + mass[j])
				newVelocity[j] = (deltaVel * mass[i] * min(C[j], C[i]) + t) / (mass[i] + mass[j])
 
	for i in range(len(newVelocity)) : 
		newVelocity[i] = np.abs(newVelocity[i]) ** 0.999 * np.sign(newVelocity[i])
	

	return newPos, newVelocity

def calcCollisionWithWall(posLst : list[np.ndarray], velocity : list[np.ndarray], C : list[float], radius : list[float], deltaTime) -> tuple[list[np.ndarray], list[np.ndarray]] :
	newPos, newVelocity = [], []
	for i in range(len(posLst)) :
		newPos.append(posLst[i])
		newVelocity.append(velocity[i])
	for i in range(len(posLst)) :
		if newPos[i][0] < radius[i] + BorderWidth or newPos[i][0] > screenWidth - radius[i] - BorderWidth :
			newVelocity[i] = np.array([C[i] * -newVelocity[i][0], newVelocity[i][1]])
			newPos[i] = (max(radius[i] + BorderWidth, min(newPos[i][0], screenWidth - radius[i] - BorderWidth)), newPos[i][1])
		if newPos[i][1] < radius[i] + BorderWidth or newPos[i][1] > screenHeight - radius[i] - BorderWidth :
			newVelocity[i] = np.array([newVelocity[i][0], C[i] * -newVelocity[i][1]])
			newPos[i] = (newPos[i][0], max(radius[i] + BorderWidth, min(newPos[i][1], screenHeight - radius[i] - BorderWidth)))
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
			curPlayerPos, targetPlayerPos = PlayerInitPos.copy(), PlayerInitPos.copy()
			curDefenderPos = np.array([screenWidth / 2, ])
			curBallPos, curBallVel = BallInitPos.copy(), np.zeros(2)
			startTime = time.monotonic()
			gameEnd = False
			needReflesh = False

		if not gameEnd :
			head = [detect_face.detect()]
			if head[0] :
				targetPlayerPos = np.array(
					[(-head[0][0] + 0.5) * screenWidth * 2 + screenWidth / 2, 
					head[0][1] * screenHeight + screenHeight / 4])
				targetPlayerPos[0] = max(PlayerRadius, min(targetPlayerPos[0], screenWidth - PlayerRadius))
				targetPlayerPos[1] = max(PlayerRadius, min(targetPlayerPos[1], screenHeight - PlayerRadius))
			
			deltaTime = time.monotonic() - curTime
			curTime = time.monotonic()
			newPlayerPos = calcPlayerPos(curPlayerPos, targetPlayerPos)
			newDefenderPos = calcDefenderPos(curDefenderPos, calcDefenderTarget(curBallPos))
			playerVel = calcPlayerVelocity(curPlayerPos, newPlayerPos, deltaTime)
			defenderVel = calcPlayerVelocity(curDefenderPos, newDefenderPos, deltaTime)

			curPos, curVel = calcCollision([curPlayerPos, curBallPos, curDefenderPos], 
								  	[playerVel, curBallVel, defenderVel],
									mass, radius, 
									[calcCofPlayer(playerVel), 1, 1], 
									deltaTime)
			curPlayerPos, curDefenderPos = newPlayerPos, newDefenderPos
			curPos, curVel = calcCollisionWithWall([curPos[1]], [curVel[1]], [1], [BallRadius], deltaTime)
			curBallPos = curPos[0]
			curBallVel = curVel[0]

			inDefendRangeNewState = checkInDefendRange(curBallPos)
			if inDefendRangeNewState :
				if not inDefendRangeLstState :
					inDefendRangeStartTime = time.monotonic()

			inDefendRangeLstState = inDefendRangeNewState

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

			
			# show remaining time
			remainTime = 30 - (time.monotonic() - startTime)
			if remainTime < 0 :
				remainTime = 0
			text = font.render("Time: " + str(int(remainTime)), True, (0, 0, 0))
			# pyg.draw.rect(screen, (255, 255, 255, 1), (screenWidth - text.get_width(), screenHeight - text.get_height() - 10, 100, text.get_height() + 10), 0)
			screen.blit(text, (screenWidth - text.get_width() - 25 - 10, screenHeight - text.get_height() - 10 - 25))

			

			if inDefendRangeLstState :
				remainTime = 10 - (time.monotonic() - inDefendRangeStartTime)
				if remainTime < 0 :
					remainTime = 0
				# text = font.render(str(int(remainTime)), True, (0, 0, 0))
				# screen.blit(text, (screenWidth / 2 - text.get_width() / 2, screenHeight - text.get_height() * 2 - 10))


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