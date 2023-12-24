class TableObject {
    constructor(encodingType = 'label') {
        this.reset(encodingType);
    }

    reset(encodingType) {
        this._encodingType = encodingType;
        this._headers = [];
        this._data = [];
        this._categoricalColumns = [];
        this._encodedColumns = {};
    }

    clear() {
        this.reset(this._encodingType);
    }

    set headers(headers) {
        if (!Array.isArray(headers)) {
            throw new Error("Headers must be of type array");
        }
        this._headers = headers;
        this.#updateCategoricalColumns();
    }

    get headers() {
        return this._headers;
    }

    set data(arrArr) {
        if (!Array.isArray(arrArr)) {
            throw new Error("Data must be of type array");
        }
        this._data = arrArr;
        this.#updateCategoricalColumns();
    }

    get data() {
        return this._data;
    }

    setDataAndHeaders(arrArr, alt = "Col") {
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
            headers = firstRow.map((_, i) => `${alt}${i + 1}`);
            data = arrArr;
        }

        // Set headers and data
        this.headers = headers;
        this.data = data;
    }

    #updateCategoricalColumns() {
        if (this._data.length > 0 && this._headers.length > 0) {
            this._categoricalColumns = this.#detectCategoricalColumns();
            this._encodedColumns = this.#prepareEncodedColumns();
        }
    }

    #detectCategoricalColumns() {
        let categoricalColumns = [];
        this.data.forEach(row => {
            row.forEach((value, index) => {
                if (isNaN(value) && !categoricalColumns.includes(this.headers[index])) {
                    categoricalColumns.push(this.headers[index]);
                }
            });
        });
        return categoricalColumns;
    }

    #prepareEncodedColumns() {
        let encodedColumns = {};

        this._categoricalColumns.forEach(column => {
            let uniqueValuesMap = new Set(this.data.map(row => row[this.headers.indexOf(column)]));

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
                let encodedColumnData = this.data.map(row => {
                    let encodedRow = [];
                    uniqueValuesMap.forEach(uniqueValue => {
                        encodedRow.push(row[this.headers.indexOf(column)] === uniqueValue ? 1 : 0);
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

    getEncodedHeaders() {
        if (this._encodingType === 'label') return this._headers;
        let encodedHeaders = [];

        this.headers.forEach(header => {
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

    getEncodedData() {
        return this.data.map(row => {
            let encodedRow = [];

            this.headers.forEach((header, headerIndex) => {
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

    decodeData(encodedData) {
        if (!Array.isArray(encodedData)) {
            throw new Error("Encoded data must be an array");
        }

        return encodedData.map(encodedRow => {
            let decodedRow = [];
            let encodedIndex = 0;

            this.headers.forEach((header, headerIndex) => {
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

    getPreviewData(encoded) {
        let headers;
        let data;
        if (encoded) {
            headers = this.getEncodedHeaders();
            data = this.getEncodedData();
        } else {
            headers = this._headers;
            data = this._data;
        }
        return {
            headers: headers,
            data: sampleArrArr(data),
        }
    }

    prepareSimilarData(encoded, encodedData) {
        let headers;
        let data;
        if (encoded) {
            headers = this.getEncodedHeaders();
            data = encodedData;
        } else {
            headers = this._headers;
            data = this.decodeData(encodedData)
        }
        return {
            headers: headers,
            data: data
        }
    }
}
