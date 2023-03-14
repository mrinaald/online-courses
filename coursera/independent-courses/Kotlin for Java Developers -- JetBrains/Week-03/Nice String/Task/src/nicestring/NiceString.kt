package nicestring

fun String.isNice(): Boolean {
    /* *** My Solution *** */
    /*
    val cond1: Boolean = ("bu" !in this) && ("ba" !in this) && ("be" !in this)

    var vowelCount = 0
    for (c in this) {
        if (c in "aeiou")
            vowelCount++
    }

    val cond2: Boolean = vowelCount >= 3
    if (cond1 && cond2)
        return true

    var cond3 = false
    for (i in 1 until this.length) {
        if (this[i] == this[i-1]) {
            cond3 = true
            break
        }
    }

    if (cond2 && cond3)
        return true

    if (cond1 && cond3)
        return true

    return false
    */

    /* Kotlin-way solution *** */
    val cond1: Boolean = setOf("bu", "ba", "be").none { this.contains(it) }

    val cond2: Boolean = this.count { it in "aeiou" } >= 3

    val cond3: Boolean = this.zipWithNext().any { it.first == it.second }

    return listOf(cond1, cond2, cond3).count { it } >= 2
}