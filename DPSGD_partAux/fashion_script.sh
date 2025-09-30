#!/bin/bash
# Loop over indices 0 to 7
for i in {0..7}
do
    echo "=============================="
    echo "Starting iteration $i"
    echo "=============================="
    echo "$i" | python -m DPSGD_partAux.fashion_eps05.py
done

for i in {0..7}
do
    echo "=============================="
    echo "Starting iteration $i"
    echo "=============================="
    echo "$i" | python -m DPSGD_partAux.fashion_eps1.py
done

for i in {0..7}
do
    echo "=============================="
    echo "Starting iteration $i"
    echo "=============================="
    echo "$i" | python -m DPSGD_partAux.fashion_eps15.py
done

for i in {0..7}
do
    echo "=============================="
    echo "Starting iteration $i"
    echo "=============================="
    echo "$i" | python -m DPSGD_partAux.fashion_eps2.py
done

for i in {0..7}
do
    echo "=============================="
    echo "Starting iteration $i"
    echo "=============================="
    echo "$i" | python -m DPSGD_partAux.fashion_eps5.py
done

for i in {0..7}
do
    echo "=============================="
    echo "Starting iteration $i"
    echo "=============================="
    echo "$i" | python -m DPSGD_partAux.fashion_eps10.py
done