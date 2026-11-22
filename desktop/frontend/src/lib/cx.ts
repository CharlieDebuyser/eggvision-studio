/** Concatène des classes CSS en ignorant les valeurs falsy : cx('btn', active && 'is-active'). */
export function cx(...parts: Array<string | false | null | undefined | 0>): string {
  return parts.filter(Boolean).join(' ');
}
