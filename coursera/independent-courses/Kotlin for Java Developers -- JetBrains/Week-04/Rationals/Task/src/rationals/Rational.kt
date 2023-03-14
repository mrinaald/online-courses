package rationals

import java.math.BigInteger

class Rational {
    private var numerator: BigInteger = BigInteger.ZERO
    private var denominator: BigInteger = BigInteger.ONE

    fun setRational(num: BigInteger, den: BigInteger = BigInteger.ONE) {
        // denominator should not be equal to zero
        require(den.compareTo(BigInteger.ZERO) != 0)

        if (num.compareTo(BigInteger.ZERO) == 0) {
            numerator = BigInteger.ZERO
            denominator = BigInteger.ONE
            return
        }

        val (n, d) = when (num.compareTo(BigInteger.ZERO)
                            to den.compareTo(BigInteger.ZERO)) {
            // both are positive
            1 to 1 -> num to den
            // both are negative
            -1 to -1 -> num.abs() to den.abs()
            // when numerator is negative
            -1 to 1 -> num to den
            // when denominator is negative
            1 to -1 -> num.negate() to den.negate()
            else -> BigInteger.ZERO to BigInteger.ONE
        }

        val g = n.gcd(d)

        numerator = n.divide(g)
        denominator = d.divide(g)
    }

    operator fun plus(other: Rational): Rational {
        val n1 = numerator.multiply(other.denominator)
        val n2 = denominator.multiply(other.numerator)

        return generateRational(n1.add(n2),
                                denominator.multiply(other.denominator))
    }

    operator fun minus(other: Rational): Rational {
        val n1 = numerator.multiply(other.denominator)
        val n2 = denominator.multiply(other.numerator)

        return generateRational(n1.subtract(n2),
                denominator.multiply(other.denominator))
    }

    operator fun times(other: Rational): Rational {
        return generateRational(numerator.multiply(other.numerator),
                                denominator.multiply(other.denominator))
    }

    operator fun div(other: Rational): Rational {
        return generateRational(numerator.multiply(other.denominator),
                                denominator.multiply(other.numerator))
    }

    operator fun unaryMinus(): Rational =
            generateRational(numerator.negate(), denominator)

    operator fun unaryPlus(): Rational =
        generateRational(numerator, denominator)

    operator fun compareTo(other: Rational): Int =
            this.toDouble().compareTo(other.toDouble())

    operator fun rangeTo(end: Rational): Pair<Double, Double> =
            this.toDouble() to end.toDouble()

    override operator fun equals(other: Any?): Boolean {
        if (other == null)
            return false

        when (other) {
            is Rational -> return (this.numerator == other.numerator) &&
                                (this.denominator == other.denominator)
        }

        return false
    }

    fun toDouble(): Double = numerator.toDouble() / denominator.toDouble()

    override fun toString(): String {
        return if (denominator == BigInteger.ONE)
            numerator.toString()
        else
            "$numerator/$denominator"
    }
}


fun generateRational(n: BigInteger = BigInteger.ZERO,
                     d: BigInteger = BigInteger.ONE): Rational {
    val rational = Rational()
    rational.setRational(n, d)
    return rational
}


infix fun Int.divBy(den: Any): Rational =
        when (den) {
            is Int, is Long -> generateRational(BigInteger(this.toString()),
                                                BigInteger(den.toString()))
            is BigInteger -> generateRational(BigInteger(this.toString()),
                                                den)
            else -> generateRational()
        }


infix fun Long.divBy(den: Any): Rational =
        when (den) {
            is Int, is Long -> generateRational(BigInteger(this.toString()),
                                                BigInteger(den.toString()))
            is BigInteger -> generateRational(BigInteger(this.toString()),
                                                den)
            else -> generateRational()
        }


infix fun BigInteger.divBy(den: Any): Rational =
        when (den) {
            is Int, is Long -> generateRational(this,
                                                BigInteger(den.toString()))
            is BigInteger -> generateRational(this, den)
            else -> generateRational()
        }


fun String.toRational(): Rational =
    if ('/' in this) {
        val n = this.split('/')
        BigInteger(n[0]) divBy BigInteger(n[1])
    }
    else {
        BigInteger(this) divBy BigInteger.ONE
    }


operator fun Pair<Double, Double>.contains(rational: Rational): Boolean {
    val d = rational.toDouble()

    return (this.first <= d) && (d <= this.second)
}


fun main() {
    val half = 1 divBy 2
    val third = 1 divBy 3

    val sum: Rational = half + third
    println(5 divBy 6 == sum)

    val difference: Rational = half - third
    println(1 divBy 6 == difference)

    val product: Rational = half * third
    println(1 divBy 6 == product)

    val quotient: Rational = half / third
    println(3 divBy 2 == quotient)

    val negation: Rational = -half
    println(-1 divBy 2 == negation)

    println((2 divBy 1).toString() == "2")
    println((-2 divBy 4).toString() == "-1/2")
    println("117/1098".toRational().toString() == "13/122")

    val twoThirds = 2 divBy 3
    println(half < twoThirds)

    println(half in third..twoThirds)

    println(2000000000L divBy 4000000000L == 1 divBy 2)

    println("912016490186296920119201192141970416029".toBigInteger() divBy
            "1824032980372593840238402384283940832058".toBigInteger() ==
            1 divBy 2)
}
