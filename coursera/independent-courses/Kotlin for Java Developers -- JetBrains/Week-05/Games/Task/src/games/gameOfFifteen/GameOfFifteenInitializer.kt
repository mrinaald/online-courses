package games.gameOfFifteen

interface GameOfFifteenInitializer {
    /*
     * Even permutation of numbers 1..15
     * used to initialized the first 15 cells on a board.
     * The last cell is empty.
     */
    val initialPermutation: List<Int>
}

class RandomGameInitializer : GameOfFifteenInitializer {
    /*
     * Generate a random permutation from 1 to 15.
     * `shuffled()` function might be helpful.
     * If the permutation is not even, make it even (for instance,
     * by swapping two numbers).
     */
    override val initialPermutation by lazy {
        val auxiliaryList = (1..15).toList().shuffled()

        if (isEven(auxiliaryList))
            return@lazy auxiliaryList
        else
            return@lazy makeEvenPermutation(auxiliaryList)

    }

    private fun makeEvenPermutation(list: List<Int>): List<Int> {
        val evenParityList = list.toMutableList()

        for (i in list.indices) {
            for (j in i+1 until list.size) {
                if (evenParityList[i] > evenParityList[j]) {
                    val temp = evenParityList[i]
                    evenParityList[i] = evenParityList[j]
                    evenParityList[j] = temp
                    return evenParityList.toList()
                }
            }
        }

        return evenParityList.toList()
    }
}

