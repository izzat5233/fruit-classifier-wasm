function toArr(vec) {
    const arr = [];
    for (let i = 0; i < vec.size(); i++) arr.push(vec.get(i));
    return arr;
}

function toVecNum(arr) {
    const vec = new Module.VecNum();
    for (let i of arr) vec.push_back(Number(i));
    return vec;
}

function toVecUInt(arr) {
    const vec = new Module.VecUInt();
    for (let i of arr) vec.push_back(Number(i));
    return vec;
}

function toArrArr(vecVec) {
    const arrArr = [];
    for (let i = 0; i < vecVec.size(); i++) arrArr.push(toArr(vecVec.get(i)));
    return arrArr;
}

function toVecVecNum(arrArr) {
    const vecVec = new Module.VecVecNum();
    for (let arr of arrArr) vecVec.push_back(toVecNum(arr));
    return vecVec;
}

function toArrArrArr(vecVecVec) {
    const arrArrArr = [];
    for (let i = 0; i < vecVecVec.size(); i++) arrArrArr.push(toArrArr(vecVecVec.get(i)));
    return arrArrArr;
}

function toVecVecVecNum(arrArrArr) {
    const vecVecVec = new Module.VecVecVecNum();
    for (let arrArr of arrArrArr) vecVecVec.push_back(toVecVecNum(arrArr));
    return vecVecVec;
}

function csvToArrArr(csvString) {
    const arrArr = [];
    const lines = csvString.split('\n');

    for (let line of lines) {
        if (line.trim() === '') continue;
        arrArr.push(line.split(','));
    }
    return arrArr;
}

function arrArrToCsv(arrArr) {
    let csvString = '';
    for (let arr of arrArr) csvString += arr.join(',') + '\n';
    return csvString;
}

function sampleArrArr(arrArr, sampleSize = 10) {
    const res = [];
    const dataSize = arrArr.length;
    const interval = Math.max(1, Math.floor(dataSize / sampleSize));

    for (let i = 0; i < dataSize && res.length < sampleSize; i += interval) {
        res.push(arrArr[i]);
    }
    return res;
}