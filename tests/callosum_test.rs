#[cfg(test)]
mod tests {
    use super::*;
    use URIELV1::Callosum::event::CallosumEvent;
    use URIELV1::Callosum::callosum::CorpusCallosum;

    #[test]
    fn test_callosum_publish_and_subscribe() {
        let callosum = CorpusCallosum::new();

        // Subscribe to events
        let rx = callosum.subscribe("test_subscriber");

        // Create and publish a sample event
        let test_message = "This is a test".to_string();
        let event = CallosumEvent::RawInput(test_message.clone());
        callosum.publish(event);

        // Receive and assert
        let received = rx.recv().expect("Did not receive event");
        match &*received {
            CallosumEvent::RawInput(s) => assert_eq!(s.as_str(), test_message),
            _ => panic!("Unexpected event type received"),
        }

    }
}
