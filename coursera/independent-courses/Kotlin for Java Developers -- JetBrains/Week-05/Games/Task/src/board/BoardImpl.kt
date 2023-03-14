package board

import board.Direction.*
import java.lang.Integer.max
import java.lang.Integer.min

open class MySquareBoard(override val width: Int): SquareBoard {
    private var board = arrayOf<Array<Cell>>()

    init {
        for (r in 1..width) {
            var row = arrayOf<Cell>()
            for (c in 1..width) {
                row += Cell(r, c)
            }
            board += row
        }
    }

    override fun getCellOrNull(i: Int, j: Int): Cell? =
            if ( (i < 1) || (i > width) || (j < 1) || (j > width) )
                null
            else
                board[i-1][j-1]


    override fun getCell(i: Int, j: Int): Cell {
        require((1 <= i) && (i <= width))
        require((1 <= j) && (j <= width))

        return board[i-1][j-1]
    }

    override fun getAllCells(): Collection<Cell> {
        return board.flatten()
    }

    override fun getRow(i: Int, jRange: IntProgression): List<Cell> {
        if ( (i < 1) || (i > width) )
            return listOf<Cell>()

        return if (jRange.first < jRange.last) {
            val first: Int = max(jRange.first-1, 0)
            val last: Int = min(width, jRange.last)
            board[i-1].toList().subList(first, last)
        }
        else {
            val first: Int = max(jRange.last-1, 0)
            val last: Int = min(width, jRange.first)
            board[i-1].toList().subList(first, last).reversed()
        }
    }

    override fun getColumn(iRange: IntProgression, j: Int): List<Cell> {
        if ( (j < 1) || (j > width) )
            return listOf<Cell>()

        val start: Int
        val end: Int
        val isReverse: Boolean

        if (iRange.first < iRange.last) {
            start = max(iRange.first, 1)
            end = min(width, iRange.last)
            isReverse = false
        }
        else {
            start = max(iRange.last, 1)
            end = min(width, iRange.first)
            isReverse = true
        }

        val column = mutableListOf<Cell>()
        for (i in start..end) {
            column += board[i-1][j-1]
        }
        return if (isReverse) column.reversed() else column
    }

    override fun Cell.getNeighbour(direction: Direction): Cell? {
        if ( (i < 1) || (i > width) || (j < 1) || (j > width) )
            return null

        return when (direction) {
            UP -> if (this.i == 1) null else board[this.i-2][this.j-1]
            DOWN -> if (this.i == width) null else board[this.i][this.j-1]
            RIGHT -> if (this.j == width) null else board[this.i-1][this.j]
            LEFT -> if (this.j == 1) null else board[this.i-1][this.j-2]
        }
    }
}

fun createSquareBoard(width: Int): SquareBoard = MySquareBoard(width)




class MyGameBoard<T>(override val width: Int): GameBoard<T> {
    private val gameBoard = mutableMapOf<Cell, T?>()
    private val board = MySquareBoard(width)

    init {
        for (c in board.getAllCells()) {
            gameBoard[c] = null
        }
    }

    override fun getCellOrNull(i: Int, j: Int): Cell? =
            board.getCellOrNull(i, j)


    override fun getCell(i: Int, j: Int): Cell =
            board.getCell(i, j)


    override fun getAllCells(): Collection<Cell> =
            board.getAllCells()


    override fun getRow(i: Int, jRange: IntProgression): List<Cell> =
            board.getRow(i, jRange)


    override fun getColumn(iRange: IntProgression, j: Int): List<Cell> =
            board.getColumn(iRange, j)

    override fun Cell.getNeighbour(direction: Direction): Cell? =
            with(board) {
                return@with this@getNeighbour
            }

    private fun getCellOrNull(cell: Cell): Cell? =
            board.getCellOrNull(cell.i, cell.j)


    override fun get(cell: Cell): T? {
        val c = getCellOrNull(cell)
        return if (c == null)
            null
        else
            gameBoard[c]
    }

    override fun set(cell: Cell, value: T?) {
        val c = getCellOrNull(cell)
        if (c != null)
            gameBoard[c] = value
    }

    override fun filter(predicate: (T?) -> Boolean): Collection<Cell> =
            gameBoard.filterValues { value -> predicate.invoke(value) }.keys


    override fun find(predicate: (T?) -> Boolean): Cell? =
            gameBoard.filterValues { v -> predicate.invoke(v) }.keys.first()

    override fun any(predicate: (T?) -> Boolean): Boolean =
            gameBoard.filterValues { v -> predicate.invoke(v) }.isNotEmpty()

    override fun all(predicate: (T?) -> Boolean): Boolean =
            gameBoard.filterValues { v -> predicate.invoke(v) }.size == gameBoard.size
}

fun <T> createGameBoard(width: Int): GameBoard<T> = MyGameBoard<T>(width)

