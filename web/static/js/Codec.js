/**
 * Data Encoder-Decoder
 */
class Codec {
    constructor() {
        this.clear();
    }

    clear() {
        this._headers = [];
        this._categoricalColumns = [];
        this._encodedColumns = {};
    }

    use(arrArr, encodingType = 'label', headerAlt = "Col") {
        if (!Array.isArray(arrArr) || arrArr.length === 0) {
            throw new Error("Invalid input: arrArr must be a non-empty array");
        }

        const firstRow = arrArr[0];
        let headers, data;

        // Detect if the first row should be treated as headers
        if (firstRow.every(val => isNaN(val))) {
            headers = firstRow;
            data = arrArr.slice(1);
        } else {
            headers = firstRow.map((_, i) => `${headerAlt}${i + 1}`);
            data = arrArr;
        }

        this._encodingType = encodingType;
        this._headers = headers;
        this._categoricalColumns = this.#detectCategoricalColumns(data);
        this._encodedColumns = this.#prepareEncodedColumns(data);
    }

    #detectCategoricalColumns(data) {
        let categoricalColumns = [];
        data.forEach(row => {
            row.forEach((value, index) => {
                if (isNaN(value) && !categoricalColumns.includes(this._headers[index])) {
                    categoricalColumns.push(this._headers[index]);
                }
            });
        });
        return categoricalColumns;
    }

    #prepareEncodedColumns(data) {
        let encodedColumns = {};

        this._categoricalColumns.forEach(column => {
            let uniqueValuesMap = new Set(data.map(row => row[this._headers.indexOf(column)]));

            if (this._encodingType === 'label') {
                let labelMap = Array.from(uniqueValuesMap).reduce((acc, val, index) => {
                    acc[val] = index;
                    return acc;
                }, {});

                encodedColumns[column] = {
                    uniqueValues: Array.from(uniqueValuesMap),
                    labelMap: labelMap
                };
            } else {
                let encodedColumnData = data.map(row => {
                    let encodedRow = [];
                    uniqueValuesMap.forEach(uniqueValue => {
                        encodedRow.push(row[this._headers.indexOf(column)] === uniqueValue ? 1 : 0);
                    });
                    return encodedRow;
                });
                encodedColumns[column] = {
                    uniqueValues: Array.from(uniqueValuesMap),
                    data: encodedColumnData
                };
            }
        });

        return encodedColumns;
    }

    getHeaders(encoded = false) {
        return encoded ? this.#getEncodedHeaders() : this._headers;
    }

    #getEncodedHeaders() {
        if (this._encodingType === 'label') return this._headers;
        let encodedHeaders = [];

        this._headers.forEach(header => {
            if (this._categoricalColumns.includes(header)) {
                this._encodedColumns[header].uniqueValues.forEach(uniqueValue => {
                    encodedHeaders.push(`${header}_${uniqueValue}`);
                });
            } else {
                encodedHeaders.push(header);
            }
        });

        return encodedHeaders;
    }

    encode(data, skipHeaders = true) {
        return data.slice(skipHeaders ? 1 : 0).map(row => {
            let encodedRow = [];

            this._headers.forEach((header, headerIndex) => {
                if (this._categoricalColumns.includes(header)) {
                    if (this._encodingType === 'label') {
                        // Label encoding
                        const labelValue = this._encodedColumns[header].labelMap[row[headerIndex]];
                        encodedRow.push(labelValue);
                    } else {
                        // One-hot encoding
                        const valueIndex = this._encodedColumns[header].uniqueValues.indexOf(row[headerIndex]);
                        const oneHotArray = Array(this._encodedColumns[header].uniqueValues.length).fill(0);
                        if (valueIndex >= 0) {
                            oneHotArray[valueIndex] = 1;
                        }
                        encodedRow = encodedRow.concat(oneHotArray);
                    }
                } else {
                    encodedRow.push(row[headerIndex]);
                }
            });

            return encodedRow;
        });
    }

    decode(data, skipHeaders = true) {
        if (!Array.isArray(data)) {
            throw new Error("Encoded data must be an array");
        }

        return data.slice(skipHeaders ? 1 : 0).map(encodedRow => {
            let decodedRow = [];
            let encodedIndex = 0;

            this._headers.forEach((header, headerIndex) => {
                if (this._categoricalColumns.includes(header)) {
                    if (this._encodingType === 'label') {
                        // Decode label encoding with rounding and clipping
                        const numberOfLabels = this._encodedColumns[header].uniqueValues.length;
                        let labelValueIndex = Math.round(encodedRow[encodedIndex]);
                        labelValueIndex = Math.max(0, Math.min(labelValueIndex, numberOfLabels - 1));
                        const decodedValue = this._encodedColumns[header].uniqueValues[labelValueIndex];
                        decodedRow.push(decodedValue);
                        encodedIndex++;
                    } else {
                        // Decode one-hot encoding
                        const numberOfCategories = this._encodedColumns[header].uniqueValues.length;
                        const encodedSegment = encodedRow.slice(encodedIndex, encodedIndex + numberOfCategories);
                        const maxProbIndex = encodedSegment.indexOf(Math.max(...encodedSegment));
                        const decodedValue = this._encodedColumns[header].uniqueValues[maxProbIndex];
                        decodedRow.push(decodedValue);
                        encodedIndex += numberOfCategories;
                    }
                } else {
                    // For non-categorical columns, the value is directly taken from the encoded data
                    decodedRow.push(encodedRow[encodedIndex]);
                    encodedIndex++;
                }
            });
            return decodedRow;
        });
    }

    get(encoded, data, isDataEncoded, skipHeaders = true) {
        let res;
        if (encoded) {
            res = isDataEncoded ? data.slice(skipHeaders ? 1 : 0) : this.encode(data, skipHeaders);
        } else {
            res = isDataEncoded ? this.decode(data, skipHeaders) : data.slice(skipHeaders ? 1 : 0);
        }
        return {
            headers: this.getHeaders(encoded),
            data: res
        }
    }
}
