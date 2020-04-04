#!/bin/bash
# Script-uri utilitare pentru reteaua neur(on)ala CitNet
# CitNet foloseste modelul EfficientNet (vezi https://arxiv.org/abs/1905.11946)

export OMP_THREAD_LIMIT=1
IFS=$'
'

 # Conversie in jpg 128x256 px
find "/home/localadmin/citnet/data_original/de antrenat/" -type f -name "*.pdf"| parallel --progress "convert xc:white[128x256\!] -respect-parenthesis \( {}[0-1] -resize 595x842\> -background white -extent 595x842 -resize 128x128! -append \) -composite -set colorspace Gray -depth 8 -strip {= s'/pdf'/dataset';s'.pdf'.jpg' =}"

 # Conversie in tif 64x128 px
#find "/home/localadmin/citnet/data_original/de antrenat/" -type f -name "*.pdf"| parallel --progress "convert xc:white[64x128\!] -respect-parenthesis \( {}[0-1] -resize 595x842\> -background white -extent 595x842 -resize 64x64! -append \) -composite -set colorspace Gray -depth 8 -strip {= s'/pdf'/dataset';s'.pdf'.tif' =}"

 # Pentru Citatii
#listafoldere=("/mnt/cagl/Prelucrate/" "/mnt/trgl/Prelucrate/" "/mnt/jdgl/Prelucrate/" "/mnt/jdtecuci/Prelucrate/" "/mnt/jdtgb/Prelucrate/" "/mnt/jdliesti/Prelucrate/" "/mnt/br/TrBraila/Prelucrate/" "/mnt/br/JdBraila/Prelucrate/" "/mnt/jdfaurei/Prelucrate/" "/mnt/trvn/Prelucrate/" "/mnt/jdfocsani/Prelucrate/" "/mnt/jdadjud/Prelucrate/" "/mnt/jdpanciu/Prelucrate/")
#numar=(4000 4000 4000 500 500 500 4000 4000 500 4000 0 500 500)
#for ((i=0;i<${#listafoldere[@]};i++))
#  do
#    find ${listafoldere[$i]} -type f -regextype posix-extended -regex '.*_[a-z0-9]{3}_[0-9]{5}\.pdf' | head -${numar[$i]} | parallel --progress -j100 -X "cp -t "/home/localadmin/pdf/dataset/training/citatii/" {}"
#  done

 # Pentru Documente
#listafoldere=("/mnt/cagldoc/Prelucrate/" "/mnt/trgl/Prelucrate/" "/mnt/jdgl/Prelucrate/" "/mnt/jdtecuci/Prelucrate/" "/mnt/jdtgb/Prelucrate/" "/mnt/jdliesti/Prelucrate/" "/mnt/br/TrBraila/Prelucrate/" "/mnt/br/JdBraila/Prelucrate/" "/mnt/jdfaurei/Prelucrate/" "/mnt/trvn/Prelucrate/" "/mnt/jdfocsani/Prelucrate/" "/mnt/jdadjud/Prelucrate/" "/mnt/jdpanciu/Prelucrate/")
#numar=(400000 400000 400000 500 500 500 4000 4000 500 4000 0 500 500)
#for ((i=0;i<${#listafoldere[@]};i++))
#  do
#    find ${listafoldere[$i]} -type f -regextype posix-extended -regex '.*[0-9]+-[1-9][0-9]*-.*\.pdf' | head -${numar[$i]} | parallel --progress -X "cp -u -t "/home/localadmin/pdf/dataset/training/documente/" {}"
#  done

 # Organizare in foldere
#find /home/localadmin/pdf/dataset/training/citatii/ -type f -name "*.pdf"| shuf -n 2000 | parallel --progress "mv -f {} {= s/training/testing/ =}"
#find /home/localadmin/pdf/dataset/training/citatii/ -type f -name "*.pdf"| shuf -n 2000 | parallel --progress "mv -f {} {= s/training/validation/ =}"
#find /home/localadmin/pdf/dataset/training/documente/ -type f -name "*.pdf"| shuf -n 2000 | parallel --progress "mv -f {} {= s/training/testing/ =}"
#find /home/localadmin/pdf/dataset/training/documente/ -type f -name "*.pdf"| shuf -n 2000 | parallel --progress "mv -f {} {= s/training/validation/ =}"

# Redimensioneaza tif la 64x128 grayscale 8-bit 
#for f in $(find /home/localadmin/citnet/dataset/ -type f -name "*.tif")
#  do
#    if [[ $(identify "$f") != *"64x128"* ]]
#	   then
#	     identify "$f"  
#		 echo "Resize: $f"
#		 convert $f -resize 64x128! -set colorspace Gray -depth 8 -strip $f
#	   fi
#  done

 # Verificare fisiere in plus
# for i in /home/localadmin/citnet/pdf/testing/documente/*.*
#  do
#    filename=${i##*/}
#    filename=${filename%.*}
#    if [ ! -f "/home/localadmin/citnet/dataset/testing/documente/$filename.tif" ]
#      then
#        echo $i
#      fi
#  done
