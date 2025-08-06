// A global variable to hold the RTCPeerConnection object.
var pc = null;

// This function handles the WebRTC negotiation process.
function negotiate() {
    // Add a video transceiver to the peer connection.
    // 'recvonly' means this client will only receive video.
    pc.addTransceiver('video', { direction: 'recvonly' });
    // Add an audio transceiver to the peer connection.
    // 'recvonly' means this client will only receive audio.
    pc.addTransceiver('audio', { direction: 'recvonly' });

    // Create a WebRTC offer (a session description).
    // This describes the client's capabilities.
    return pc.createOffer().then((offer) => {
        // Set the created offer as the local description.
        // This makes the offer available to the remote peer.
        return pc.setLocalDescription(offer);
    }).then(() => {
        // Wait for ICE (Interactive Connectivity Establishment) gathering to complete.
        // This process finds all possible network addresses (candidates) for the connection.
        return new Promise((resolve) => {
            // Check if ICE gathering is already complete.
            if (pc.iceGatheringState === 'complete') {
                // If so, resolve the promise immediately.
                resolve();
            } else {
                // If not, create a function to check the state.
                const checkState = () => {
                    // Check if ICE gathering is now complete.
                    if (pc.iceGatheringState === 'complete') {
                        // If so, remove the event listener to prevent it from firing again.
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        // Resolve the promise, allowing the next step in the chain to run.
                        resolve();
                    }
                };
                // Add an event listener to call `checkState` whenever the ICE gathering state changes.
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        // Once ICE gathering is complete, get the final offer with all the candidates.
        var offer = pc.localDescription;
        // Use the Fetch API to send the offer to a server.
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        // The server's response should contain the remote peer's answer.
        return response.json();
    }).then((answer) => {
        // Set the value of an HTML element with the ID 'sessionid' to the session ID from the answer.
        document.getElementById('sessionid').value = answer.sessionid
        // Set the received answer as the remote description.
        // This tells the local peer connection about the remote peer's capabilities.
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        alert(e);
    });
}

// This function initiates the WebRTC connection.
function start() {
    // Define the configuration for the RTCPeerConnection.
    var config = {
        // Specify 'unified-plan' SDP semantics, which is the modern standard.
        sdpSemantics: 'unified-plan'
    };

    // Check if the 'use-stun' checkbox is checked.
    if (document.getElementById('use-stun').checked) {
        // If so, add a STUN server to the configuration.
        // STUN (Session Traversal Utilities for NAT) helps discover the public IP address and port.
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    // Create a new RTCPeerConnection with the defined configuration.
    pc = new RTCPeerConnection(config);

    // Add an event listener for the 'track' event.
    // This event fires when a new media track is received from the remote peer.
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            // If it's video, set the source of the video element to the received stream.
            document.getElementById('video').srcObject = evt.streams[0];
        } else {
            // Otherwise (if it's audio), set the source of the audio element to the received stream.
            document.getElementById('audio').srcObject = evt.streams[0];
        }
    });

    // Hide the 'start' button.
    document.getElementById('start').style.display = 'none';
    // Call the negotiation function to begin the connection process.
    negotiate();
    // Show the 'stop' button.
    document.getElementById('stop').style.display = 'inline-block';
}

// This function handles closing the WebRTC connection.
function stop() {
    // Hide the 'stop' button.
    document.getElementById('stop').style.display = 'none';

    // Wait 500 milliseconds before closing the peer connection.
    // This gives time for any final data to be sent/received.
    setTimeout(() => {
        // Close the RTCPeerConnection.
        pc.close();
    }, 500);
}

// This event handler is triggered when the window is unloaded (e.g., page is closed or reloaded).
window.onunload = function (event) {
    // Wait 500 milliseconds before closing the peer connection.
    setTimeout(() => {
        pc.close();
    }, 500);
};

// This event handler is triggered before the window is unloaded.
// It can be used to show a confirmation prompt to the user.
window.onbeforeunload = function (e) {
    // Wait 500 milliseconds before closing the peer connection.
    setTimeout(() => {
        pc.close();
    }, 500);
    // Get the event object, compatible with different browsers.
    e = e || window.event
    // For older IE and Firefox, set the returnValue property.
    if (e) {
        e.returnValue = 'Close prompt';
    }
    // For modern browsers, return a string which is displayed in the prompt.
    return 'Close prompt';
}