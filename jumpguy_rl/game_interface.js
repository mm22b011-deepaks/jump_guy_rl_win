// Minimal bridge - just provides screenshot and input dispatch
(function() {
    if (window.__rlBridge) return 'already_loaded';

    window.__rlBridge = {
        isReady: function() {
            return document.querySelector('canvas') !== null;
        },

        getScreenshot: function() {
            const canvas = document.querySelector('canvas');
            if (!canvas) return null;
            try {
                return canvas.toDataURL('image/png');
            } catch(e) {
                return null;
            }
        },

        // Dispatch keyboard event on canvas
        sendKey: function(keyCode, code, key) {
            const canvas = document.querySelector('canvas');
            if (!canvas) return false;
            canvas.focus();

            canvas.dispatchEvent(new KeyboardEvent('keydown', {
                bubbles: true, cancelable: true,
                key: key, code: code,
                keyCode: keyCode, which: keyCode, charCode: keyCode,
            }));
            return true;
        },

        // Dispatch click/tap on canvas
        sendClick: function(x, y) {
            const canvas = document.querySelector('canvas');
            if (!canvas) return false;
            canvas.focus();
            x = x || 480;
            y = y || 270;

            const opts = {
                bubbles: true, cancelable: true,
                clientX: x, clientY: y,
                screenX: x, screenY: y,
                button: 0, buttons: 1,
            };

            canvas.dispatchEvent(new PointerEvent('pointerdown', { ...opts, pointerId: 1, pointerType: 'mouse' }));
            canvas.dispatchEvent(new MouseEvent('mousedown', opts));
            canvas.dispatchEvent(new PointerEvent('pointerup', { ...opts, pointerId: 1, pointerType: 'mouse', buttons: 0 }));
            canvas.dispatchEvent(new MouseEvent('mouseup', { ...opts, buttons: 0 }));
            canvas.dispatchEvent(new MouseEvent('click', { ...opts, buttons: 0 }));
            return true;
        }
    };

    return 'loaded';
})();
