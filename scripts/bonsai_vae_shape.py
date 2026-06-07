TOKEN_GRID=4
PATCH_SIZE=2

def token_count():
    return TOKEN_GRID*TOKEN_GRID

def token_dim(ch):
    return ch*PATCH_SIZE*PATCH_SIZE
