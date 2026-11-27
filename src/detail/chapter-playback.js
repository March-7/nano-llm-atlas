// Scene drawing and the transport share this clock. Scrubbing and pausing never
// accumulate hidden time, and a superseded chapter cannot advance the tour.
export const phaseMillis = 2550;

export function chapterPlayback({
  onFrame,
  onComplete,
  now = () => performance.now(),
  schedule = (fn) => requestAnimationFrame(fn),
  cancel = (id) => cancelAnimationFrame(id),
}) {
  let phases = [],
    elapsed = 0,
    since = 0,
    frame = null,
    status = 'idle',
    revision = 0;

  function position() {
    return Math.min(
      phases.length,
      elapsed + (status === 'playing' ? Math.max(0, now() - since) / phaseMillis : 0),
    );
  }
  function snapshot() {
    if (!phases.length) return null;
    const at = position();
    const index = Math.min(Math.floor(at), phases.length - 1);
    return {
      phase: phases[index],
      index,
      count: phases.length,
      progress: at - index,
      position: at,
      status,
    };
  }
  function unschedule() {
    revision++;
    if (frame !== null) cancel(frame);
    frame = null;
  }
  function queue() {
    const ticket = revision;
    frame = schedule(() => {
      if (revision !== ticket) return;
      frame = null;
      if (position() >= phases.length) {
        elapsed = phases.length;
        status = 'ended';
        onFrame(snapshot());
        if (revision === ticket) onComplete();
      } else {
        onFrame(snapshot());
        if (revision === ticket && status === 'playing') queue();
      }
    });
  }
  function play() {
    if (!phases.length || status === 'playing') return;
    if (status === 'ended') elapsed = 0;
    since = now();
    status = 'playing';
    onFrame(snapshot());
    queue();
  }
  function pause() {
    if (status !== 'playing') return;
    elapsed = position();
    unschedule();
    status = elapsed === phases.length ? 'ended' : 'paused';
    onFrame(snapshot());
  }
  return {
    get state() {
      return snapshot();
    },
    start(next, autoplay = true) {
      unschedule();
      phases = next;
      elapsed = 0;
      status = phases.length ? 'paused' : 'idle';
      if (autoplay && phases.length) play();
      else onFrame(snapshot());
    },
    refresh(next) {
      if (!phases.length || next.length !== phases.length) return;
      phases = next;
      onFrame(snapshot());
    },
    play,
    pause,
    seek(at) {
      if (!phases.length || !Number.isFinite(at)) return;
      unschedule();
      elapsed = Math.max(0, Math.min(phases.length, at));
      status = elapsed === phases.length ? 'ended' : 'paused';
      onFrame(snapshot());
    },
  };
}
