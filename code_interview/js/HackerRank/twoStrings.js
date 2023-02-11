function twoStrings (s1,s2) {
    let output = "NO"
    for(let i=0; i<s1.length; i++) {
        if (s2.includes(s1[i])) {
            output = 'YES'
        }
    }
    return output
}

console.log(twoStrings('hello', 'world'))