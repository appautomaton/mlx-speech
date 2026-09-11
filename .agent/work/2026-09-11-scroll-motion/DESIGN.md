# Scroll-driven presentation

Use native CSS view timelines behind a feature query and a no-preference motion
query. The hero supplies a shared timeline so its console drifts independently
of the normal document scroll. Section headings and the quickstart console use
their own entry timelines. Animate transforms only; all content stays visible.

Reduce travel on stacked layouts. Each entering element settles before reading
or interaction; keyboard focus removes motion from the relevant console.
Browsers without view timelines and visitors requesting reduced motion retain
the existing layout. No scroll listeners, animation library, or media hooks.
