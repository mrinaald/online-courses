package taxipark

/*
 * Task #1. Find all the drivers who performed no trips.
 */
fun TaxiPark.findFakeDrivers(): Set<Driver> =
        allDrivers.filter { d -> trips.none { t -> t.driver == d } }.toSet()

/*
 * Task #2. Find all the clients who completed at least the given number of trips.
 */
fun TaxiPark.findFaithfulPassengers(minTrips: Int): Set<Passenger> =
        allPassengers.filter { p -> trips.count { t -> p in t.passengers } >= minTrips }.toSet()

/*
 * Task #3. Find all the passengers, who were taken by a given driver more than once.
 */
fun TaxiPark.findFrequentPassengers(driver: Driver): Set<Passenger> =
        trips.filter { it.driver == driver }.flatMap { it.passengers }.groupBy { it }.filter { it.value.size > 1 }.map { it.key }.toSet()

/*
 * Task #4. Find the passengers who had a discount for majority of their trips.
 */
fun TaxiPark.findSmartPassengers(): Set<Passenger> =
        allPassengers.filter { p ->
            val (noDiscount, discount) = trips.filter { t -> p in t.passengers }.map { t -> t.discount }.partition { it == null }
            discount.size > noDiscount.size
        }.toSet()

/*
 * Task #5. Find the most frequent trip duration among minute periods 0..9, 10..19, 20..29, and so on.
 * Return any period if many are the most frequent, return `null` if there're no trips.
 */
fun TaxiPark.findTheMostFrequentTripDurationPeriod(): IntRange? {
    if (trips.isEmpty())
        return null

    val maxDurations = trips.map { ((it.duration/10)*10)..(((it.duration/10)*10)+9) }.groupBy { it }.maxBy { it.value.size }

    return maxDurations?.key
}

/*
 * Task #6.
 * Check whether 20% of the drivers contribute 80% of the income.
 */
fun TaxiPark.checkParetoPrinciple(): Boolean {
    if (trips.isEmpty())
        return false

    val numTopDrivers = (allDrivers.size * 0.2).toInt()
    val totalIncome = trips.fold(0.0) { sum, trip -> sum + trip.cost }

    val topIncomes = trips.groupBy({ it.driver }, { it.cost }).map { it.value.sum() }.sortedByDescending { it }.slice(0 until numTopDrivers)

    return (topIncomes.sum() >= (totalIncome*0.8))
}