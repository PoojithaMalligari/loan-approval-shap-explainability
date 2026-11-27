// Drop into your React project and render under your prediction result:
//   <ReasonsList reasons={result.top_reasons} />
export default function ReasonsList({ reasons = [] }) {
  if (!reasons.length) return null;
  return (
    <div style={{ marginTop: 16 }}>
      <h4>Why this decision?</h4>
      <ul>
        {reasons.map((r) => (
          <li key={r.feature}>
            <b>{r.feature}</b> ({r.value}) — {r.effect}
          </li>
        ))}
      </ul>
    </div>
  );
}
