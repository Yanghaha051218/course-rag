# The pebble counter

The course's imaginary pebble counter starts at zero. Each blue tick adds two
pebbles. Each amber tick removes one pebble, but the counter never goes below
zero.

After exactly three blue ticks, the counter writes a checkpoint before any later
ticks are processed. A reset clears both the pebble count and the checkpoint.
