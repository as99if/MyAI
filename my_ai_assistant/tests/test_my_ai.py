import os
import sys
import pygame


if __name__ == "__main__":
    # pygame.init()
    # screen = pygame.display.set_mode((400, 300))
    # pygame.display.set_caption("Test Window")

    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
        
    #     screen.fill((0, 0, 0))
    #     pygame.display.flip()

    # pygame.quit()
    # sys.exit()
    from src.core.my_ai import MyAI
    my_ai = MyAI(is_gui_enabled=True)
    my_ai.__run__()