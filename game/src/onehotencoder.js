const numValues = 3;
const decodedSize = 9;

function encode(array) {
  let arrSize = array.length;
  encArray = new Array(arrSize * numValues).fill(0);

  for (let i = 0, j; i < arrSize; i++) {
    j = i * 3;
    encArray[j + array[i]] = 1;
  }
  return encArray;
}

function decode(array) {
  decArray = new Array(decodedSize);
  for (let i = 0, j; i < decodedSize; i++) {
    j = i * 3;
    if (array[j] == 1) decArray[i] = 0;
    else if (array[j + 1] == 1) decArray[i] = 1;
    else if (array[j + 2] == 1) decArray[i] = 2;
  }

  return decArray;
}

module.exports = { encode, decode };
