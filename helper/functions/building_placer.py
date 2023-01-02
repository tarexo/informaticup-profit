

def get_all_mines_positions(env, deposit):
    mine_positions = []
    for subtype in range(4):
        positions = get_mine_positions_subtype(deposit, subtype)
        for x,y in positions:
            success = check_mine_position(env, x,y, subtype)
            if success: mine_positions.append([x,y,subtype])
    return mine_positions

def check_mine_position(env, x,y, subtype):
    width = env.width-1
    height = env.height-1
    if subtype == 0:
        if y>height or y<0 or x>width or x<0: return False
        field = env.grid[y][x]
        if field != ' ': return False
        if y>height or y<0 or x+1>width or x+1<0: return False
        field = env.grid[y][x+1]
        if field != ' ': return False
        for i in range(x-1, x+2):
            if y+1>height or y+1<0 or i>width or i<0: return False
            field = env.grid[y+1][i]
            if field != ' ': return False
        return True
    if subtype == 1:
        if y>height or y<0 or x+1>width or x+1<0: return False
        field = env.grid[y][x+1]
        if field != ' ': return False
        if y+1>height or y+1<0 or x+1>width or x+1<0: return False
        field = env.grid[y+1][x+1]
        if field != ' ': return False
        for i in range(y-1, y+2):
            if i>height or i<0 or x>width or x<0: return False
            field = env.grid[i][x]
            if field != ' ': return False
        return True
    if subtype == 2:
        if y+1>height or y+1<0 or x>width or x<0: return False
        field = env.grid[y+1][x]
        if field != ' ': return False
        if y+1>height or y+1<0 or x+1>width or x+1<0: return False
        field = env.grid[y+1][x+1]
        if field != ' ': return False
        for i in range(x-1, x+2):
            if y>height or y<0 or i>width or i<0: return False
            field = env.grid[y][i]
            if field != ' ': return False
        return True 
    if subtype == 3:
        if y>height or y<0 or x>width or x<0: return False
        field = env.grid[y][x]
        if field != ' ': return False
        if y+1>height or y+1<0 or x>width or x<0: return False
        field = env.grid[y+1][x]
        if field != ' ': return False
        for i in range(y-1, y+2):
            if i>height or i<0 or x+1>width or x+1<0: return False
            field = env.grid[i][x+1]
            if field != ' ': return False
        return True
    return False

def get_mine_positions_subtype(deposit, subtype:int):
    x = deposit.x
    y = deposit.y
    width = x+deposit.width
    height = y+deposit.height
    pos = []
    if subtype == 0:
        for i in range(x+1, width+1):
            pos.append([i,y-2])
        for j in range(y-1,height-1):
            pos.append([width+1,j])
        pos.append([width,height-1])
        return pos
    if subtype == 1:
        for i in range(y+1,height+1):
            pos.append([width,i])
        for j in range(x,width):
            pos.append([j,height+1])
        pos.append([x-1,height])
        return pos
    if subtype == 2:
        for i in range(x-2,width-2):
            pos.append([i,height])
        for j in range(y, height):
            pos.append([x-3,j])
        pos.append([x-2,y-1])
        return pos
    if subtype == 3:
        for i in range(y-2,height-2):
            pos.append([x-2,i])
        for j in range(x-1,width-1):
            pos.append([j,y-3])
        pos.append([width-1,y-2])
        return pos
    return None

def get_all_factory_positions(env, all_deposits):
    factory_positions = []
    width = env.width-4
    height = env.height -4
    depostis_outlets = get_deposits_outlets(all_deposits)
    for i in range(width):
        for j in range(height):
            success_position = check_factory_position(env, i, j)
            success_legal = check_next_to_deposit(env,i,j,depostis_outlets)
            if success_position and success_legal: factory_positions.append([i,j])

    return factory_positions

def check_factory_position(env, i, j):
    for x in range(i,i+5):
        for y in range(j, j+5):
            field = env.grid[y][x]
            if field != ' ': return False
    return True

def check_next_to_deposit(env,x,y, depostis_outlets):
    border = []
    for i in range(x-1, x+6):
        border.append([i,y-1])
        border.append([i,y+6])
    for i in range(y, y+5):
        border.append([x-1,i])
        border.append([x+6,i])
    for b in border:
        if b in depostis_outlets: return False
    return True

def get_deposits_outlets(deposits):
    outlets = []
    for deposit in deposits:   
        x = deposit.x
        y = deposit.y
        width = deposit.width
        height = deposit.height
        for i in range(x, x+width):
            outlets.append([i,y])
            outlets.append([i,y+height])
        for i in range(y+1, y+height-1):
            outlets.append([x,i])
            outlets.append([x+width,i])
    return outlets

