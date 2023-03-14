package games.gameOfFifteen

import board.Cell
import board.Direction
import board.createGameBoard
import games.game.Game

/*
 * Implement the Game of Fifteen (https://en.wikipedia.org/wiki/15_puzzle).
 * When you finish, you can play the game by executing 'PlayGameOfFifteen'.
 */
fun newGameOfFifteen(initializer: GameOfFifteenInitializer = RandomGameInitializer()): Game =
    GameOfFifteen(initializer)

class GameOfFifteen(private val initializer: GameOfFifteenInitializer): Game {
    private val board = createGameBoard<Int?>(4)
    private val winList: MutableList<Int?> = (1..15).toMutableList()

    init {
        winList.add(null)
    }

    override fun initialize() {
        with (board) {
            for (r in 1..width) {
                for (c in 1..width) {
                    this[getCell(r, c)] = initializer.initialPermutation
                                            .getOrNull(((r-1)*width) + (c-1))
                }
            }
        }
    }

    override fun canMove(): Boolean = true  // because one cell is always null

    override fun hasWon(): Boolean {
        for ( r in 1..board.width) {
            for (c in 1..board.width) {
                if (this[r, c] != winList[((r-1)*board.width) + (c-1)])
                    return false
            }
        }

        return true
    }

    override fun processMove(direction: Direction) {
        when (direction) {
            Direction.UP -> {
                val columnWithNull = getColumnWithNull()
                for (c in 0 until columnWithNull.size-1) {
                    if (board[columnWithNull[c]] == null) {
                        board[columnWithNull[c]] = board[columnWithNull[c+1]]
                        board[columnWithNull[c+1]] = null
                        break
                    }
                }
            }

            Direction.DOWN -> {
                val columnWithNull = getColumnWithNull()
                for (c in 1 until columnWithNull.size) {
                    if (board[columnWithNull[c]] == null) {
                        board[columnWithNull[c]] = board[columnWithNull[c-1]]
                        board[columnWithNull[c-1]] = null
                        break
                    }
                }
            }

            Direction.LEFT -> {
                val rowWithNull = getRowWithNull()
                for (r in 0 until rowWithNull.size-1) {
                    if (board[rowWithNull[r]] == null) {
                        board[rowWithNull[r]] = board[rowWithNull[r+1]]
                        board[rowWithNull[r+1]] = null
                        break
                    }
                }
            }

            Direction.RIGHT -> {
                val rowWithNull = getRowWithNull()
                for (r in 1 until rowWithNull.size) {
                    if (board[rowWithNull[r]] == null) {
                        board[rowWithNull[r]] = board[rowWithNull[r-1]]
                        board[rowWithNull[r-1]] = null
                        break
                    }
                }
            }
        }
    }

    private fun getColumnWithNull(): List<Cell> {
        val nullCell = board.find { it == null }
        return board.getColumn(1..board.width, nullCell!!.j).toList()
    }

    private fun getRowWithNull(): List<Cell> {
        val nullCell = board.find { it == null }
        return board.getRow(nullCell!!.i, 1..board.width).toList()
    }

    override fun get(i: Int, j: Int): Int? =
            board.run {
                get(getCell(i, j))
            }

}