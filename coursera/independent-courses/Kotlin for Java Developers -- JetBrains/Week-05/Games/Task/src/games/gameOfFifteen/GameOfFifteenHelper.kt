package games.gameOfFifteen

fun merge(arr: MutableList<Int>, start: Int, mid: Int, end: Int): Int {
    var i = start
    var j = mid+1
    val tempList = mutableListOf<Int>()
    var inversionCount = 0

//    print("s: $start  m: $mid  e: $end  iL: $arr  ")

    while ((i <= mid) && (j <= end)) {
        if (arr[i] > arr[j]) {
            tempList += arr[j++]
            inversionCount += mid - i +1
        }
        else {
            tempList += arr[i++]
        }
    }

    while (i <= mid)
        tempList += arr[i++]

    while (j <= end)
        tempList += arr[j++]

    tempList.forEachIndexed {index, element ->
        arr[start + index] = element
    }

//    println("sL: $arr  invs: $inversionCount")
    return inversionCount
}

fun countInversions(arr: MutableList<Int>, start: Int, end: Int): Int {
    if (start >= end)
        return 0

    val mid = start + ((end-start)/2)

    val leftInversions = countInversions(arr, start, mid)
    val rightInversions = countInversions(arr, mid+1, end)
    val mergeInversions = merge(arr, start, mid, end)

    return leftInversions + rightInversions + mergeInversions
}


/*
 * This function should return the parity of the permutation.
 * true - the permutation is even
 * false - the permutation is odd
 * https://en.wikipedia.org/wiki/Parity_of_a_permutation

 * If the game of fifteen is started with the wrong parity, you can't get the correct result
 *   (numbers sorted in the right order, empty cell at last).
 * Thus the initial permutation should be correct.
 */
fun isEven(permutation: List<Int>): Boolean {
    return countInversions(permutation.toMutableList(), 0, permutation.size-1) % 2 == 0
}