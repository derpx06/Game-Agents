import pygame,sys
from pygame import mixer

pygame.init() #Initializing pygame 
screen = pygame.display.set_mode((800,600)) #Creating Screen (width,height)
ball = pygame.image.load('ball.png') #Loading Ball Image
icon = pygame.image.load('pinball.png')
paddle1 = pygame.image.load('Paddle.png') #Loading Paddle Image
paddle2 = pygame.image.load('Paddle.png')
mixer.music.load('background.ogg')
mixer.music.play(-1)
pygame.display.set_icon(icon) #Setting Icon On title Bar
pygame.display.set_caption('Pong')
# Ball Co-ordinates
ball_x = 0
ball_y = 0
ball_dx = 0.3
ball_dy = 0.3
# Paddle1 Co-ordinates
paddle1_x = 15
paddle1_y = 200
change1 =0
# paddle 2 movements
paddle2_x = 775
paddle2_y = 200
change2 = 0
l = 0
score_value1 = 0
score_value2 = 0
score1 = pygame.font.Font('freesansbold.ttf',100)
score2 = pygame.font.Font('freesansbold.ttf',100)
game_over_text = pygame.font.Font('freesansbold.ttf',70)
play_again_txt = pygame.font.Font('freesansbold.ttf',35)

def draw_ball(x,y):          # Drawing ball On Screen
    screen.blit(ball, (x,y)) 

def draw_paddle1(x,y):
    screen.blit(paddle1, (x,y))

def draw_paddle2(x,y):
    screen.blit(paddle2, (x,y))

def show_text1():
    t = score1.render(str(score_value1),True,(255,0,0))
    screen.blit(t, (200,250))

def show_text2():
    t = score2.render(str(score_value2),True,(0,0,255))
    screen.blit(t, (550,250))

def game_over():
    global score_value2,score_value1,l
    screen.fill((0,0,0))
    pygame.draw.line(screen,(255,0,0),(0,0),(400,0),10)
    pygame.draw.line(screen,(255,0,0),(0,0),(0,600),10)
    pygame.draw.line(screen,(255,0,0),(0,600),(400,600),10)
    pygame.draw.line(screen,(0,0,255),(400,0),(800,0),10)
    pygame.draw.line(screen,(0,0,255),(800,0),(800,600),10)
    pygame.draw.line(screen,(0,0,255),(400,600),(800,600),10)    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    t = game_over_text.render('GAME OVER!',True,(23, 8, 189))
    screen.blit(t, (170,240))
    f = play_again_txt.render('Press Enter To Play Again.',True,(23, 8, 189))
    screen.blit(f, (170,350))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    l = event.key
        
        if l == pygame.K_RETURN:
            l = 0
            score_value1 = 0
            score_value2 = 0
            main()
            break    
    

def main():
    global ball_x,ball_y,ball_dx,ball_dy,paddle1_x,paddle1_y,change1
    global paddle2_x,paddle2_y,change2,score_value1,score_value2
    while True:
        screen.fill((0,0,0))
        pygame.draw.line(screen,(255,0,0),(0,0),(400,0),10)
        pygame.draw.line(screen,(255,0,0),(0,0),(0,600),10)
        pygame.draw.line(screen,(255,0,0),(0,600),(400,600),10)
        pygame.draw.line(screen,(0,0,255),(400,0),(800,0),10)
        pygame.draw.line(screen,(0,0,255),(800,0),(800,600),10)
        pygame.draw.line(screen,(0,0,255),(400,600),(800,600),10)
        pygame.draw.line(screen,(255,255,255),(400,0),(400,600),7)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    change2 = -0.5
                if event.key == pygame.K_DOWN:
                    change2 = 0.5
                if event.key == pygame.K_w:
                    change1 = -0.5
                if event.key == pygame.K_s:
                    change1 = 0.5
                
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    change2 = 0
                if event.key == pygame.K_DOWN:
                    change2 = 0
                if event.key == pygame.K_w:
                    change1 = 0
                if event.key == pygame.K_s:
                    change1 = 0

        if paddle1_y <= 0:
            paddle1_y = 0
        
        if paddle1_y >= 459:
            paddle1_y = 459 
        
        if paddle2_y <= 0:
            paddle2_y = 0
        
        if paddle2_y >= 459:
            paddle2_y = 459

        ball_x += ball_dx
        ball_y += ball_dy

        if ball_y >= 584:
            ball_y = 584
            ball_dy = -0.3
        
        if ball_y <= 0:
            ball_y = 0
            ball_dy = 0.3

        if (ball_x<=14) and (int(ball_y) in list(range(int(paddle1_y), int(paddle1_y+141)))):
            s = mixer.Sound('pong.ogg')
            mixer.Sound.play(s)
            ball_dx = 0.3
        
        if (ball_x>=760) and (int(ball_y) in list(range(int(paddle2_y), int(paddle2_y+141)))):
            e = mixer.Sound('pong.ogg')
            mixer.Sound.play(e)
            ball_dx = -0.3


        if ball_x<=-17:
            score_value2+=1 
            pygame.time.delay(1000)
            ball_x=400
            ball_y=300

        if ball_x>=801:
            score_value1+=1
            pygame.time.delay(1000)
            ball_x=400
            ball_y=300
        
        if score_value2 > 10 or score_value1 > 10:
            game_over()
            break
        
        paddle1_y += change1
        paddle2_y += change2
        draw_ball(ball_x,ball_y)
        draw_paddle1(paddle1_x,paddle1_y)
        draw_paddle2(paddle2_x,paddle2_y)
        show_text1()
        show_text2()
        pygame.display.update() #Updating The Display

main()
