import cython

"""
Takes in board, returns (value, location)
"""
cpdef (double, int) minimax(board, int depth, bint maximizingPlayer, double alpha, double beta, dict seen):
    cdef double discountFactor = 0.99
    cdef int check = check_win(board)
    cdef double maxEval = -20
    cdef int maxPlace = -1 
    cdef double minEval = 20 # rewards can be at most 10 or -10 so -20 equivalent to -inf
    cdef int minPlace = -1 
    cdef int[7] children = [3, 4, 2, 5, 1, 6, 0]
    cdef int[42] newBoard
    cdef double eval
    cdef int evalPlace
    cdef double returnEval 
    cdef int returnPlace
    cdef str strBoard

    if depth == 0 or check:
        if check == 1: 
            returnEval = 10
            returnPlace = -1
        elif check == 2:
            returnEval = -10
            returnPlace = -1
        else:
            returnEval = 0
            returnPlace = -1
        return (returnEval, returnPlace)
    if maximizingPlayer:
        for child in children:
            if board[child]!=0: 
                continue
            newBoard = copyArray(board)
            make_move(child, newBoard, 1)
            strBoard = str(newBoard)
            if strBoard in seen:
                eval = seen[strBoard]
            else:
                eval, evalPlace = minimax(newBoard, depth-1, False, alpha, beta, seen)
                seen[strBoard] = eval
            if eval > maxEval:
                maxEval = eval
                maxPlace = child
            alpha = max(alpha, eval)
            if beta<=alpha:
                break
        returnEval = discountFactor*maxEval
        returnPlace = maxPlace
        return (returnEval, returnPlace)
    else:

        for child in children:
            if board[child]!=0: 
                continue
            newBoard = copyArray(board)
            make_move(child, newBoard, 2)
            strBoard = str(newBoard)
            if strBoard in seen: 
                eval = seen[strBoard]
            else:
                eval, evalPlace = minimax(newBoard, depth-1, True, alpha, beta, seen)
                seen[strBoard] = eval
            if eval < minEval:
                minEval = eval
                minPlace = child
            beta = min(beta, eval)
            if beta<= alpha:
                break
        return (discountFactor*minEval, minPlace)
    
cpdef int check_win(board): 
    cdef int filled = 0
    for r in range(5, -1, -1): 
        for c in range(7): 
            if board[oneD(r,c)]!= 0:
                filled+=1
                if r>2: # checks vertical
                    win = True
                    for i in range(1, 4): 
                        if board[oneD(r-i, c)]!= board[oneD(r,c)]: 
                            win = False
                            break
                    if win: 
                        return (board[oneD(r,c)])
                if c<4: # checks horizontal
                    win = True
                    for i in range(1, 4): 
                        if board[oneD(r,c+i)]!= board[oneD(r,c)]: 
                            win = False
                            break
                    if win: 
                        return (board[oneD(r,c)])
                if r>2 and c<4: # checks upward diagonal
                    win = True
                    for i in range(1, 4): 
                        if board[oneD(r-i,c+i)]!= board[oneD(r,c)]: 
                            win = False
                            break
                    if win: 
                        return (board[oneD(r,c)])
                if r<3 and c<4: # checks downward diagonal
                    win = True
                    for i in range(1, 4): 
                        if board[oneD(r+i,c+i)]!= board[oneD(r,c)]: 
                            win = False
                            break
                    if win: 
                        return (board[oneD(r,c)])
    if filled == 42: 
        return 3

cdef int oneD(int r, int c): 
    return r*7 + c

def copyArray(arr):
    cdef int newArray[42]
    for i in range(42): 
        newArray[i] = arr[i]
    return newArray
        
cdef void make_move(int place, int board[], int turn): 
    for r in range(6): 
        if board[oneD(5-r,place)] == 0: 
            if turn == 1: 
                board[oneD(5-r,place)] = 1 #O makes first move
                turn +=1
            else: 
                board[oneD(5-r,place)] = 2
                turn  -=1
            break