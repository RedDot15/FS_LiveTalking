/*
WAV encoder + encoding engine
https://github.com/xiangyuecn/Recorder

Of course, it is best to use mp3 and wav formats, and the code also gives priority to these two formats
Browser support
https://developer.mozilla.org/en-US/docs/Web/HTML/Supported_media_formats

Encoding principle: add a 44-byte wav header to the pcm data to make a wav file; the pcm data is the original data (resampled) in the Recorder buffers, and it is in LE little-endian mode (Little Endian) at 16 bits, which is essentially not encoded at all.
*/
(function () {
	"use strict";

	Recorder.prototype.enc_wav = {
		stable: true
		, testmsg: "Supports 8-bit and 16-bit depth (filled in the bit rate), and the sample rate has no limit"
	};
	Recorder.prototype.wav = function (res, True, False) {
		var This = this, set = This.set
			, size = res.length
			, sampleRate = set.sampleRate
			, bitRate = set.bitRate == 8 ? 8 : 16;

		//Encoding data https://github.com/mattdiamond/Recorderjs https://www.cnblogs.com/blqw/p/3782420.html https://www.cnblogs.com/xiaoqi/p/6993912.html
		var dataLength = size * (bitRate / 8);
		var buffer = new ArrayBuffer(44 + dataLength);
		var data = new DataView(buffer);

		var offset = 0;
		var writeString = function (str) {
			for (var i = 0; i < str.length; i++, offset++) {
				data.setUint8(offset, str.charCodeAt(i));
			};
		};
		var write16 = function (v) {
			data.setUint16(offset, v, true);
			offset += 2;
		};
		var write32 = function (v) {
			data.setUint32(offset, v, true);
			offset += 4;
		};

		/* RIFF identifier */
		writeString('RIFF');
		/* RIFF chunk length */
		write32(36 + dataLength);
		/* RIFF type */
		writeString('WAVE');
		/* format chunk identifier */
		writeString('fmt ');
		/* format chunk length */
		write32(16);
		/* sample format (raw) */
		write16(1);
		/* channel count */
		write16(1);
		/* sample rate */
		write32(sampleRate);
		/* byte rate (sample rate * block align) */
		write32(sampleRate * (bitRate / 8));// *1 channel
		/* block align (channel count * bytes per sample) */
		write16(bitRate / 8);// *1 channel
		/* bits per sample */
		write16(bitRate);
		/* data chunk identifier */
		writeString('data');
		/* data chunk length */
		write32(dataLength);
		// Write sample data
		if (bitRate == 8) {
			for (var i = 0; i < size; i++, offset++) {
				//It is said that the 16-to-8 conversion is by Lei Xiaohua https://blog.csdn.net/sevennight1989/article/details/85376149 The details are clearer than blqw's proportional algorithm, although there are obvious noises
				var val = (res[i] >> 8) + 128;
				data.setInt8(offset, val, true);
			};
		} else {
			for (var i = 0; i < size; i++, offset += 2) {
				data.setInt16(offset, res[i], true);
			};
		};


		True(new Blob([data.buffer], { type: "audio/wav" }));
	}
})();