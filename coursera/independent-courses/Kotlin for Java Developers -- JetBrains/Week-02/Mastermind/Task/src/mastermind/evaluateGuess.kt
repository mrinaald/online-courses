package mastermind

data class Evaluation(val rightPosition: Int, val wrongPosition: Int)

fun evaluateGuess(secret: String, guess: String): Evaluation {
    var rightPosition = 0
    var wrongPosition = 0
    val countArray = Array(6){ 0 }

    secret.forEachIndexed {
        idx, s -> countArray[s - 'A'] += 1
                  if (secret[idx] == guess[idx]) rightPosition++
    }

    for (g in guess) {
        if (g in secret && countArray[g - 'A'] > 0) {
            wrongPosition++
        }
        countArray[g - 'A'] -= 1
    }

    wrongPosition -= rightPosition

    return Evaluation(rightPosition, wrongPosition)
}
