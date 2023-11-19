#!/bin/bash

# Define the list of names
name_list=("T1_CA_T2_CA" "T1_CA_T2_CC" "T1_CA_T2_E1" "T1_CA_T2_E2" "T1_CA_T2_E3" "T1_CA_T2_E4" "T1_CC_T2_CA" "T1_E1_T2_CA" "T1_E1_T2_CC" "T1_E1_T2_E1" "T1_E1_T2_E2" "T1_E1_T2_E3" "T1_E1_T2_E4" "T1_E2_T2_CA" "T1_E2_T2_CC" "T1_E2_T2_E1" "T1_E2_T2_E2" "T1_E2_T2_E3" "T1_E2_T2_E4" "T1_E3_T2_CA" "T1_E3_T2_CC" "T1_E3_T2_E1" "T1_E3_T2_E2" "T1_E3_T2_E3" "T1_E3_T2_E4")

# Rename files
count=1
for file in *".CSV"; do
    base_name=$(basename "$file" ".CSV")
    new_name=$(printf "%02d-%s" "$count" "$(echo "$base_name" | tr ' ' '_')")
    mv "$file" "$new_name.CSV"
    echo "Renamed: $file -> $new_name.CSV"
    ((count++))
done

