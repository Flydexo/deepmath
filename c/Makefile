CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm
TARGET = ai.exe

all: $(TARGET)

$(TARGET): main.o read-image.o read-label.o micrograd.o
	$(CC) -o $(TARGET) main.o read-image.o read-label.o micrograd.o $(LDFLAGS)

main.o: mnist/main.c mnist/read-image.h mnist/read-label.h micrograd/micrograd.h
	$(CC) $(CFLAGS) -c mnist/main.c

read-image.o: mnist/read-image.c mnist/read-image.h
	$(CC) $(CFLAGS) -c mnist/read-image.c

read-label.o: mnist/read-label.c mnist/read-label.h
	$(CC) $(CFLAGS) -c mnist/read-label.c

micrograd.o: micrograd/micrograd.c micrograd/micrograd.h
	$(CC) $(CFLAGS) -c micrograd/micrograd.c

clean:
	rm -f *.o $(TARGET)
