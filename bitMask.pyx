import cython


cpdef (double, int) convertAndCall(board, int depth, bint maximizingPlayer, double alpha, double beta, dict seen, int player):
    cdef long position = 0
    cdef long mask = 0
    cdef long one = 1
    cdef long oneMask
    
    for i in range(0, 6):
        for j in range(0, 7):
            oneMask = one << (j*7 + (5-i))
            if board[i][j] != 0: 
                mask |= oneMask
            if board[i][j] == player:
                position |= oneMask

    return minimax(position, mask, depth, maximizingPlayer, alpha, beta, seen)

"""
Takes in board, returns (value, location)
"""
cdef (double, int) minimax(long position, long mask, int depth, bint maximizingPlayer, double alpha, double beta, dict seen):
    cdef double discountFactor = 0.99
    cdef int win = checkWin(position)
    cdef double maxEval = -20
    cdef int maxPlace = -1 
    cdef double minEval = 20 # rewards can be at most 10 or -10 so -20 equivalent to -inf
    cdef int minPlace = -1 
    cdef int[7] children = [3, 4, 2, 5, 1, 6, 0]
    cdef tuple newBoard
    cdef double eval
    cdef int evalPlace
    cdef double returnEval 
    cdef int returnPlace
    cdef long full = 2**42-1

    if depth == 0 or win or mask == full:
        if win and maximizingPlayer: 
            returnEval = 10
            returnPlace = -1
        elif win and not maximizingPlayer:
            returnEval = -10
            returnPlace = -1
        else:
            returnEval = 0
            returnPlace = -1
        return (returnEval, returnPlace)
    if maximizingPlayer:
        for child in children:
            if filled(mask, child): 
                continue
            newPosition, newMask = bitMaskMakeMove(position, mask, child)
            newBoard = (newPosition, newMask, depth)
            if newBoard in seen:
                eval = seen[newBoard]
            else:
                eval, evalPlace = minimax(newPosition, newMask, depth-1, False, alpha, beta, seen)
                seen[(newBoard)] = eval
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
            if filled(mask, child): 
                continue
            newPosition, newMask = bitMaskMakeMove(position, mask, child)
            newBoard = (newPosition, newMask, depth)
            if newBoard in seen: 
                eval = seen[newBoard]
            else:
                eval, evalPlace = minimax(newPosition, newMask, depth-1, True, alpha, beta, seen)
                seen[newBoard] = eval
            if eval < minEval:
                minEval = eval
                minPlace = child
            beta = min(beta, eval)
            if beta<= alpha:
                break
        return (discountFactor*minEval, minPlace)


cdef int filled(long mask, int col):
    cdef long one = 1
    cdef long shiftedBit = one << (col*7+5)
    return (shiftedBit & mask)!=0

cdef int checkWin(long position):
    cdef long temp
    #horizontal check
    temp = position & (position>>7)
    if temp & (temp>>14):
        return 1
    # diagonal 1 check
    temp = position & (position>>6)
    if temp & (temp>>12):
        return 1
    
    #diagonal 2 check
    temp = position & (position>>8)
    if temp & (temp>>16):
        return 1

    #vertical check 
    temp = position & (position>>1)
    if temp & (temp>>2):
        return 1
    
    return 0
    
cdef (long, long) bitMaskMakeMove(long position, long mask, int col):
    cdef long newPosition = position ^ mask # opponent bit just inverse of position. Not affected by current move mask
    cdef long one = 1 #overflow glitch if don't do this
    cdef long newMask = mask | (mask + (one<<col*7))
    return (newPosition, newMask)