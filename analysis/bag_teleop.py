"""Read human base-teleoperation episodes out of the deployment rosbags.

Why this exists: `/cmd_vel` (autonomous) and `/cmd_vel_teleop` (human) are merged
into a single `cmd_vel.csv` by scripts/nav_diag_logger.py with no source tag, so
the navlogs cannot tell the two apart. The triggered bags under
`log/system_logs/*.bag` record them as separate topics, so every message on
`/cmd_vel_teleop` is unambiguously the CR driving the base -- whether she got
there from the nav-adjust prompt, from a navigation skill handing over, or as a
Done -> "Move base" detour out of an arm-teleop session.

The bags are uncompressed rosbag v2.0. This module parses the index and the
per-chunk IndexData records to recover message timestamps only; it never
deserializes a message, so no ROS install and no message definitions are needed.

Parsing every bag is slow, so results are cached against (path, size, mtime).
"""

import datetime
import json
import pathlib
import struct

# Silence longer than this starts a new driving episode. 15 s, not 5: the CR
# pauses mid-drive to look at where the base has got to, and at 5 s that split
# one approach to the sink on day 8 into two "episodes" 11.6 s apart. The gap
# distribution across the deployment is strongly bimodal -- that 11.6 s pause is
# the only inter-episode gap below 100 s, and the next smallest is 103 s -- so
# anything in 12-60 s merges the pause and leaves every genuinely separate
# intervention split. Day 13's two approaches to the dining table stay separate
# at 750 s apart, as they should.
EPISODE_GAP_S = 15.0


def _read_header(f):
    (hlen,) = struct.unpack("<I", f.read(4))
    raw = f.read(hlen)
    fields, i = {}, 0
    while i < len(raw):
        (flen,) = struct.unpack("<I", raw[i:i + 4])
        i += 4
        kv = raw[i:i + flen]
        i += flen
        k, _, v = kv.partition(b"=")
        fields[k.decode()] = v
    return fields


def _parse_fields(raw):
    out, i = {}, 0
    while i < len(raw):
        (n,) = struct.unpack("<I", raw[i:i + 4])
        i += 4
        k, _, v = raw[i:i + n].partition(b"=")
        i += n
        out[k.decode()] = v
    return out


def _open_index(f):
    """Seek to the index section; return (connections, chunk_positions, span).

    connections maps conn id -> (topic, callerid). The callerid is what
    separates the two publishers on /cmd_vel_teleop: /rosbridge_websocket is the
    CR driving from the webapp, /shared_autonomy_teleop is the researcher's Xbox
    controller used to place the base during setup and teardown.
    """
    assert f.read(13) == b"#ROSBAG V2.0\n", "not a rosbag v2.0 file"
    fields = _read_header(f)
    (dlen,) = struct.unpack("<I", f.read(4))
    f.read(dlen)
    f.seek(struct.unpack("<Q", fields["index_pos"])[0])

    conns, chunks, lo, hi = {}, [], None, None
    while True:
        try:
            fields = _read_header(f)
            (dlen,) = struct.unpack("<I", f.read(4))
            data = f.read(dlen)
        except struct.error:
            break
        if not fields:
            break
        op = fields.get("op", b"\x00")[0]
        if op == 0x07:                                  # connection
            conns[struct.unpack("<I", fields["conn"])[0]] = (
                fields["topic"].decode(),
                _parse_fields(data).get("callerid", b"?").decode())
        elif op == 0x06:                                # chunk info
            chunks.append(struct.unpack("<Q", fields["chunk_pos"])[0])
            s = struct.unpack("<II", fields["start_time"])
            e = struct.unpack("<II", fields["end_time"])
            s, e = s[0] + s[1] / 1e9, e[0] + e[1] / 1e9
            lo = s if lo is None else min(lo, s)
            hi = e if hi is None else max(hi, e)
    return conns, sorted(set(chunks)), (lo, hi)


def bag_span(path):
    """(first, last) message epoch in the bag, or None if it carries nothing."""
    with open(path, "rb") as f:
        _, _, span = _open_index(f)
    return span if span and span[1] else None


def topic_stamps(path, topic="/cmd_vel_teleop", callerid=None):
    """Sorted epochs of messages on `topic`, optionally from one publisher only."""
    with open(path, "rb") as f:
        conns, chunks, _ = _open_index(f)
        targets = {c for c, (t, cid) in conns.items()
                   if t == topic and (callerid is None or cid == callerid)}
        if not targets:
            return []
        out = []
        for pos in chunks:
            f.seek(pos)
            try:
                _read_header(f)                          # chunk record header
                (dlen,) = struct.unpack("<I", f.read(4))
                f.seek(dlen, 1)                          # skip payload
                # IndexData records (op=0x04) trail each chunk, one per conn.
                while True:
                    here = f.tell()
                    fields = _read_header(f)
                    (dlen,) = struct.unpack("<I", f.read(4))
                    data = f.read(dlen)
                    if fields.get("op", b"\x00")[0] != 0x04:
                        f.seek(here)
                        break
                    if struct.unpack("<I", fields["conn"])[0] in targets:
                        for i in range(0, len(data), 12):
                            secs, nsecs, _ = struct.unpack("<III", data[i:i + 12])
                            out.append(secs + nsecs / 1e9)
            except struct.error:
                continue
    return sorted(out)


def episodes(stamps, gap=EPISODE_GAP_S):
    """Group timestamps into [start, end] driving episodes."""
    eps = []
    for t in stamps:
        if not eps or t - eps[-1][1] > gap:
            eps.append([t, t])
        else:
            eps[-1][1] = t
    return eps


def scan(bag_dir, cache_path, topic="/cmd_vel_teleop", callerid=None, verbose=True):
    """{bag path: {"span": [lo, hi], "stamps": [...]}} for every bag, cached."""
    bag_dir, cache_path = pathlib.Path(bag_dir), pathlib.Path(cache_path)
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    out, parsed = {}, 0
    for bag in sorted(bag_dir.glob("*.bag")):
        st = bag.stat()
        key = f"{bag.name}:{st.st_size}:{int(st.st_mtime)}:{topic}:{callerid}"
        if key not in cache:
            span = bag_span(bag)
            cache[key] = {"span": list(span) if span else None,
                          "stamps": topic_stamps(bag, topic, callerid)}
            parsed += 1
            if verbose:
                print(f"  [bag] parsed {bag.name} "
                      f"({len(cache[key]['stamps'])} {topic} msgs)")
        out[str(bag)] = cache[key]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    if verbose and parsed:
        print(f"  [bag] {parsed} newly parsed, {len(out) - parsed} cached")
    return out


def date_of(epoch):
    return datetime.datetime.fromtimestamp(epoch).date().isoformat()
