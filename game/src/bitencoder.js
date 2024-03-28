function encode(array) {
  let enc_bits = 0;
  for (let i = 0; i < array.length; i++) {
    enc_bits = enc_bits | (array[i] << (i * 2));
  }
  return enc_bits;
}

function decode(bits, arraySize) {
  let dec_array = new Array(arraySize);
  let twobit = 0b11;
  for (let i = 0; i < arraySize; i++) {
    let i2 = i * 2;
    dec_array[i] = (bits & (twobit << i2)) >> i2;
  }
  return dec_array;
}

module.exports = { encode, decode };
