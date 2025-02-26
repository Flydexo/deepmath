CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm
TARGET = ai.exe

all: $(TARGET)

$(TARGET): main.o read-image.o read-label.o
	$(CC) -o $(TARGET) main.o read-image.o read-label.o $(LDFLAGS)

main.o: main.c read-image.h read-label.h
	$(CC) $(CFLAGS) -c main.c

read-image.o: read-image.c read-image.h
	$(CC) $(CFLAGS) -c read-image.c

read-label.o: read-label.c read-label.h
	$(CC) $(CFLAGS) -c read-label.c

clean:
	rm -f *.o $(TARGET)
