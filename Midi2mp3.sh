#!/usr/bin/env bash


SOUNDFONT="$1"
TMPDIR=./.

if [[ ! -f $SOUNDFONT ]]
then
    echo "Couldn't find the soundfont: $SOUNDFONT"
    exit 1
fi


if [ "$#" -eq 0 ]
then
    echo "usage: midi2mp3 file1.mid [file2.mid, file3.mid, ...]"
    exit 0
else
    for filename in "$2"
    do
        echo "${filename}"
        WAVFILE="$TMPDIR/${filename%.*}"

        fluidsynth -F "${WAVFILE}" $SOUNDFONT "${filename}" && \
            lame "${WAVFILE}" && \
            rm "${WAVFILE}"
    done
fi
