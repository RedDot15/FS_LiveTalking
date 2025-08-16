/*
pcm encoder + encoding engine
https://github.com/xiangyuecn/Recorder

Encoding principle: The pcm format data output by this encoder is actually the raw data of the buffers in Recorder (after resampling). For 16-bit, it is in LE (Little Endian) mode and has not undergone any encoding processing.

The encoding code is not much different from wav.js. A pcm file with a 44-byte wav header becomes a wav file. So, playing pcm is very simple, just convert it to a wav file to play. The conversion function Recorder.pcm2wav is already provided.
*/
(function () {
	"use strict";

	Recorder.prototype.enc_pcm = {
		stable: true
		, testmsg: "pcm is raw unencapsulated audio data, pcm data files cannot be played directly; supports 8-bit and 16-bit (filled in bitrate), unlimited sampling rate values"
	};
	Recorder.prototype.pcm = function (res, True, False) {
		var This = this, set = This.set
			, size = res.length
			, bitRate = set.bitRate == 8 ? 8 : 16;

		var buffer = new ArrayBuffer(size * (bitRate / 8));
		var data = new DataView(buffer);
		var offset = 0;

		// Write sample data
		if (bitRate == 8) {
			for (var i = 0; i < size; i++, offset++) {
				// The 16 to 8 conversion is said to be from Lei Xiaohua's https://blog.csdn.net/sevennight1989/article/details/85376149 The details are clearer than blqw's proportional algorithm, although both have obvious noise
				var val = (res[i] >> 8) + 128;
				data.setInt8(offset, val, true);
			};
		} else {
			for (var i = 0; i < size; i++, offset += 2) {
				data.setInt16(offset, res[i], true);
			};
		};
		True(new Blob([data.buffer], { type: "audio/pcm" }));
	};
	
	/**
	Directly transcode pcm to wav, which can be played directly; requires wav.js to be included at the same time
	data: {
		sampleRate:16000 pcm's sample rate
		bitRate:16 pcm's bit depth, value: 8 or 16
		blob:blob object
		}
		If data is a directly provided blob, it will default to 16-bit 16khz configuration, for testing purposes only
	True(wavBlob,duration)
	False(msg)
	**/
	Recorder.pcm2wav = function (data, True, False) {
		if (data.slice && data.type != null) {//Blob for testing
			data = { blob: data };
		};
		var sampleRate = data.sampleRate || 16000, bitRate = data.bitRate || 16;
		if (!data.sampleRate || !data.bitRate) {
			console.warn("pcm2wav must provide sampleRate and bitRate");
		};
		if (!Recorder.prototype.wav) {
			False("pcm2wav must first load the wav encoder wav.js");
			return;
		};

		var reader = new FileReader();
		reader.onloadend = function () {
			var pcm;
			if (bitRate == 8) {
				// 8-bit to 16-bit
				var u8arr = new Uint8Array(reader.result);
				pcm = new Int16Array(u8arr.length);
				for (var j = 0; j < u8arr.length; j++) {
					pcm[j] = (u8arr[j] - 128) << 8;
				};
			} else {
				pcm = new Int16Array(reader.result);
			};

			Recorder({
				type: "wav"
				, sampleRate: sampleRate
				, bitRate: bitRate
			}).mock(pcm, sampleRate).stop(function (wavBlob, duration) {
				True(wavBlob, duration);
			}, False);
		};
		reader.readAsArrayBuffer(data.blob);
	};



})();