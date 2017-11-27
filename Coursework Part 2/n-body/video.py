import sys
import ctypes
from sdl2 import *
from sdl2.sdlgfx import *

import rdata

def main():
    renderIdx = 0
    renderLen = len(rdata.data)
    
    SDL_Init(SDL_INIT_VIDEO)
    window = SDL_CreateWindow(
        b"Hello World",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        1000,1000,
        SDL_WINDOW_SHOWN
    )
    renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_ACCELERATED|SDL_RENDERER_PRESENTVSYNC
    )

    running = True
    event = SDL_Event()
    while running:
        while SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == SDL_QUIT:
                running = False
                break
            if event.type == SDL_KEYDOWN:
                if event.key.keysym.sym == SDLK_q: running = False; break;
                if event.key.keysym.sym == SDLK_SPACE: running = False; break;
                if event.key.keysym.sym == SDLK_RETURN: running = False; break;
                if event.key.keysym.sym == SDLK_ESCAPE: running = False; break;

        SDL_SetRenderDrawColor(renderer, 0,0,0, 255)
        SDL_RenderClear(renderer);

        print renderIdx % renderLen
        for body in rdata.data[renderIdx % renderLen]:
            filledCircleRGBA(
                renderer,
                body[0], body[1], body[2],
                255,0,255, 255
            )
        SDL_RenderPresent(renderer)
        renderIdx += 1

    SDL_DestroyWindow(window)
    SDL_Quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())