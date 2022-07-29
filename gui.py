import pygame
import time
import numpy as np
import random
from env import TicTacEnv
from stable_baselines3 import DQN


class Graphics:
    '''Initializes all game graphics'''
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Tic-Tac-Toe")
        self.width, self.height = (400, 400)
        self.screen = pygame.display.set_mode((self.width, self.height+100),0,32)
        self.line_color = (10,10,10)
        self.__load_images()
        self.__game_opening()
        
    def __game_opening(self):
        self.screen.blit(self.opening,(0,0))
        pygame.display.update()
        time.sleep(1)
        self.screen.fill((255, 255, 255))
        # Drawing vertical lines
        pygame.draw.line(self.screen, self.line_color,(self.width/3,0),(self.width/3, self.height),7)
        pygame.draw.line(self.screen, self.line_color,(self.width/3*2,0),(self.width/3*2, self.height),7)
        # Drawing horizontal lines
        pygame.draw.line(self.screen, self.line_color,(0,self.height/3),(self.width, self.height/3),7)
        pygame.draw.line(self.screen, self.line_color,(0,self.height/3*2),(self.width, self.height/3*2),7)
    
    def __load_images(self):
        self.opening = scale_image(pygame.image.load("tic tac opening.png"),1)
        self.x_draw = pygame.transform.scale(pygame.image.load("X.png"),(80,80))
        self.o_draw = pygame.transform.scale(pygame.image.load("O.png"),(80,80))
        
    def xo_images(self):
        return self.x_draw, self.o_draw
        
    def get_size(self) -> tuple:
        return self.width, self.height
    
def scale_image(img: pygame.image, factor: float) -> pygame.image:
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)


def get_available(grid: list[list[int]]) -> list[tuple[int, int]]:
    available = []
    rows, cols = grid.shape[0], grid.shape[1]
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                available.append((i, j))
    return available
    
class Game:
    
    def __init__(self):
        self.player_x = DQN.load("ppo_x_model") #AI playing as X
        self.player_o = DQN.load("ppo_o_model") #AI playing as O
        self.reset_game()
        self.turn_map = {0: "tie", 1: "X", -1: "O"}
        self.screen = self.game_init.screen
        self.width, self.height = self.game_init.get_size()
        self.x_img, self.o_img = self.game_init.xo_images()
        
        
    def run(self):
        while not self.exit:
            pygame.display.update()
            #self.play_step()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
            time.sleep(0.5)
            self.cpu_x()
            time.sleep(0.5)
            self.rand_cpu()
            if(self.winner or self.draw):
                self.reset_game()
            self.clock.tick(60)
            
    def play_step(self):
        '''Human vs AI'''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # the user clicked; place an X or O
                self.on_click()
                time.sleep(1)
                self.cpu_o()
                if(self.winner or self.draw):
                    self.reset_game()
                        
    def draw_status(self):
        if self.winner:
            message = f"{self.winner} won the game!"
        else:
            message = f"It is {self.turn_map[self.turn]}'s Turn"
            
        if self.draw:
            message = 'Game Draw!'
            
        font = pygame.font.Font(None, 30)
        text = font.render(message, 1, (255, 255, 255))
        # copy the rendered message onto the board
        self.screen.fill ((0, 0, 0), (0, 400, 500, 100))
        text_rect = text.get_rect(center=(self.width/2, 500-50))
        self.screen.blit(text, text_rect)
        pygame.display.update()
    
    def check_winner(self):
        is_full = False
        available = get_available(self.board)
        score = self.env.check_winner(self.board.reshape(9))
        if score == 0 or score is None:
            pass
        elif score == 1:
            self.winner = "X"
        else:
            self.winner = "O"
        if score == 0 and len(available) == 0:
            is_full = True
        if is_full and self.winner is None:
            self.draw = True
        self.draw_status()
        
    def on_click(self):
        x, y = pygame.mouse.get_pos()
        #get column of mouse click (1-3)
        if(x < self.width/3):
            col = 1
        elif (x < self.width/3*2):
            col = 2
        elif(x < self.width):
            col = 3
        else:
            col = None
        #get row of mouse click (1-3)
        if(y < self.height/3):
            row = 1
        elif (y < self.height/3*2):
            row = 2
        elif(y < self.height):
            row = 3
        else:
            row = None
        if((row and col) and self.board[row-1][col-1] == 0):
            #draw the x or o on screen
            self.draw_xo(row,col)
            self.check_winner() 
            
    def draw_xo(self,row,col):
        if row==1:
            posx = 30
        if row==2:
            posx = self.width/3 + 30
        if row==3:
            posx = self.width/3*2 + 30
        if col==1:
            posy = 30
        if col==2:
            posy = self.height/3 + 30
        if col==3:
            posy = self.height/3*2 + 30
        
        self.board[row-1][col-1] = self.turn
        if(self.turn == 1):
            self.screen.blit(self.x_img,(posy,posx))
            self.turn = -1
        else:
            self.screen.blit(self.o_img,(posy,posx))
            self.turn = 1
        pygame.display.update()    
        
    def cpu_x(self):
        """AI trained as X player"""
        action = self.player_x.predict(self.board)[0]
        row, col = action
        self.draw_xo(row+1, col+1)
        self.check_winner()
            
    def cpu_o(self):
        """AI trained as O player"""
        action = self.player_o.predict(self.board, deterministic=True)[0]
        row, col = action
        self.draw_xo(row+1, col+1)
        self.check_winner()
            
    def rand_cpu(self):
        '''AI that plays random moves'''
        avail: list[tuple[int, int]] = get_available(self.board)
        if len(avail) > 0:
            move = random.choice(avail)
            to_play = move[0]+1, move[1]+1
            self.draw_xo(to_play[0], to_play[1])
            self.check_winner()
        
    def reset_game(self):
        time.sleep(1)
        self.board = np.zeros((3, 3))
        self.clock = pygame.time.Clock()
        self.env = TicTacEnv()
        self.turn = 1
        self.winner = None
        self.draw = False
        self.exit = False
        self.game_init = Graphics()
        
        
def main():
    game = Game()
    game.run()
    
if __name__ == '__main__':
    main()