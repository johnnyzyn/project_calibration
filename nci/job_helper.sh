#!/bin/bash

LOSSES=("focal" "mmce")
# ARCH_NUMS = ("14643" "12503" "8413" "8924" "4921" "11674" "14859" "3654" "10213" "11939" "8373" "7019" "3258" "11927" "2517" "1395" "8917" "11836" "4663" "7997" "8197" "13799" "2980" "1110" "10487" "4104" "3335" "5996" "2604" "12779" "3521" "8385" "2764" "13063" "5131" "10589" "10240" "2362" "6750" "1573" "14551" "12445" "14576" "4452" "8200" "12013" "3445" "14183" "8033" "5878" "3712" "2867" "10501" "5368" "3378" "1762" "11420" "3244" "10504" "2612" "3683" "6605" "4050" "12834" "2684" "8075" "669" "10243" "14110" "4912" "7360" "10576" "7747" "4644" "14883" "535" "6529" "10480" "6759" "7038" "451" "14023" "5396" "13849" "1735" "14700" "13117" "1815" "1740" "8569" "796" "4825" "12520" "14412" "996" "2190" "3211" "6786" "314" "4491")

for LOSS in "${LOSSES[@]}"
  do
    for ARCH_NUM in "${ARCH_NUMS[@]}"
      do
        qsub -N "${LOSS}-${ARCH_NUM}" -v LOSS="${LOSS}",ARCH_NUM="${ARCH_NUM}" job.sh
      done
  done

