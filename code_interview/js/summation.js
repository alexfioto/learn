function summation(n) {
    let output = 0
    for (i=n; n>=0; i--) {
        output += n
        n--
    }
    return output
}

console.log(summation(3))
