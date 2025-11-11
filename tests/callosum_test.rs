use std::time::Duration;

use URIELV1::callosum::callosum::CorpusCallosum;
use URIELV1::callosum::event::CallosumEvent;
use URIELV1::callosum::event::test_support::TestDynPayload;

#[test]
fn pub_sub_basic_and_fanout() {
    let bus = CorpusCallosum::new();
    let rx_a = bus.subscribe("A");
    let rx_b = bus.subscribe("B");

    let corr = Some(42);
    let id = bus.publish("test_src", corr, CallosumEvent::RawInput("hello".into()));

    // No cloning; just read the Arc<EventEnvelope>
    let env_a = rx_a.recv_timeout(Duration::from_millis(500)).expect("A recv");
    let env_b = rx_b.recv_timeout(Duration::from_millis(500)).expect("B recv");

    assert_eq!(env_a.id, id);
    assert_eq!(env_b.id, id);
    assert_eq!(env_a.source, "test_src");
    assert_eq!(env_b.source, "test_src");
    assert_eq!(env_a.correlation_id, corr);
    assert_eq!(env_b.correlation_id, corr);

    match (&env_a.event, &env_b.event) {
        (CallosumEvent::RawInput(a), CallosumEvent::RawInput(b)) => {
            assert_eq!(a, "hello");
            assert_eq!(b, "hello");
        }
        _ => panic!("expected RawInput events"),
    }
}

#[test]
fn dynamic_payload_roundtrip() {
    let bus = CorpusCallosum::new();
    let rx = bus.subscribe("t");

    bus.publish("test", None, CallosumEvent::dynamic(TestDynPayload { val: 7 }));

    let env = rx.recv_timeout(Duration::from_millis(500)).expect("recv");

    match &env.event {
        CallosumEvent::Dynamic(b) => {
            let p = b.as_any()
                .downcast_ref::<TestDynPayload>()
                .expect("downcast TestDynPayload");
            assert_eq!(p.val, 7);
        }
        other => panic!("expected Dynamic payload, got {:?}", other),
    }
}
