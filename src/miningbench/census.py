"""Read-only SWI store census; never enumerate or copy stored row terms."""

from pathlib import Path
import resource
import time


def store_census(step):
    import janus_swi

    started = time.perf_counter()
    rows = janus_swi.query_once(
        "findall([_Name,_Arity,_Count,_Bytes], "
        "(current_predicate(_Name/_Arity), atom(_Name), "
        "functor(_Head,_Name,_Arity), predicate_property(_Head,dynamic), "
        "predicate_property(_Head,number_of_clauses(_Count)), "
        "predicate_property(_Head,size(_Bytes))), Rows)"
    )["Rows"]
    stores = [
        dict(name=name, arity=arity, rows=count, bytes=size)
        for name, arity, count, size in rows
    ]
    stores.sort(key=lambda row: row["bytes"], reverse=True)
    stats = janus_swi.query_once(
        "statistics(globalused,Global), statistics(localused,Local), "
        "statistics(trailused,Trail), statistics(heapused,Heap)"
    )
    stats.pop("truth", None)
    rss_kib = int(
        next(
            line.split()[1]
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
    )
    return {
        "step": step,
        "stores": stores,
        "statistics_bytes": stats,
        "rss_kib": rss_kib,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "census_seconds": time.perf_counter() - started,
        "method": "predicate_property number_of_clauses/size; no row materialization or GC",
    }
